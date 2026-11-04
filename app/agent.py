"""Bounded StudyPlan Agent controller. System owns validation and stopping."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from app.config import current_time, settings
from app.llm import (
    InvalidModelOutputError,
    ModelUnavailableError,
    OutputBudgetExceededError,
    ModelUsage,
    decide,
)
from app.models import (
    WEEKDAYS,
    AgentDecision,
    AgentState,
    ArenaRequest,
    ArenaResponse,
    ClassSlot,
    Metrics,
    StudySession,
    StudyTask,
    ToolTrace,
    UnavailableSlot,
)
from app.prompts import build_messages
from app.tools import TOOLS

log = logging.getLogger("studyplan.agent")

# Evidence must name something specific; "soon" or "a while" can never ground a value.
SPECIFIC_DATE = re.compile(
    r"\d|\b(?:today|tonight|tomorrow|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|jan|january|feb|february|"
    r"mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|"
    r"nov|november|dec|december)\b",
    re.IGNORECASE,
)
SPECIFIC_HOURS = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half|an hour)\b",
    re.IGNORECASE,
)


class SemanticValidationError(ValueError):
    """Raised when a decision is schema-valid but semantically illegal."""


@dataclass
class RunLog:
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None

    def add_usage(self, usage: ModelUsage) -> None:
        if usage.input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + usage.input_tokens
        if usage.output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + usage.output_tokens
        if usage.estimated_cost_usd is not None:
            self.cost = (self.cost or 0) + usage.estimated_cost_usd


async def run_agent(request: ArenaRequest, history: list[BaseMessage] | None, model: str) -> ArenaResponse:
    history = history or []
    state = AgentState(
        request_id=request.request_id,
        goal=request.task.strip(),
        max_steps=min(request.arena_config.max_steps, settings.max_steps),
        preferred_hours={"start": settings.preferred_start, "end": settings.preferred_end},
        source_text=_source_text(request, history),
    )
    fault = request.arena_config.fault.type
    run = RunLog()
    invalid_in_a_row = 0
    repair_note: str | None = None

    log.info("run_start request_id=%s max_steps=%s fault=%s model=%s", state.request_id, state.max_steps, fault, model)

    while True:
        if state.step_count >= state.max_steps:
            return _finish(
                state,
                run,
                status="budget_exceeded",
                stop_reason="max_steps_exceeded",
                final_response=(
                    "Stopped because the step budget was exhausted before the plan could be finished."
                    if not repair_note
                    else "Stopped while repairing an invalid decision: step budget exhausted."
                ),
            )

        state.step_count += 1
        step = state.step_count
        force_invalid = fault == "invalid_agent_decision" and not state.fault_triggered
        messages = build_messages(state, history, request.external_context, repair_note=repair_note)
        raw_decision: dict[str, Any] | None = None

        try:
            state.model_calls += 1
            raw_decision, usage = await decide(state, messages, model_name=model, force_invalid=force_invalid)
            state.fault_triggered = state.fault_triggered or force_invalid
            run.add_usage(usage)
            decision = AgentDecision.model_validate(raw_decision)
            _semantic_validate(decision, state)
        except OutputBudgetExceededError as exc:
            run.add_usage(exc.usage)
            run.errors.append({"step": step, "type": "output_budget", "detail": str(exc)})
            return _finish(
                state,
                run,
                status="budget_exceeded",
                stop_reason="output_token_budget_exceeded",
                final_response=(
                    "Stopped because this request needs more model output than the configured token budget. "
                    "Please split it into fewer tasks."
                ),
            )
        except ModelUnavailableError as exc:
            log.warning("model_unavailable step=%s detail=%s", step, exc)
            run.errors.append({"step": step, "type": "model_unavailable", "detail": str(exc)})
            run.events.append({"step": step, "event": "model_unavailable"})
            decision = _forced_decision(state)
            if decision is None:
                return _finish(
                    state,
                    run,
                    status="failed",
                    stop_reason="model_unavailable",
                    final_response="The language model is temporarily unavailable, so I stopped without guessing. Please retry shortly.",
                )
            run.events.append({"step": step, "event": "deterministic_fallback", "action": decision.action})
        except Exception as exc:  # noqa: BLE001 - schema, semantic and JSON failures share one repair policy
            if isinstance(exc, InvalidModelOutputError):
                run.add_usage(exc.usage)
            state.repair_attempts += 1
            invalid_in_a_row += 1
            detail = str(exc)[:500]
            run.errors.append({"step": step, "type": "contract_error", "detail": detail, "raw": str(raw_decision)[:300]})
            run.events.append({"step": step, "event": "decision_rejected", "detail": detail, "repair": invalid_in_a_row})
            log.info("decision_rejected step=%s repair=%s detail=%s", step, invalid_in_a_row, detail)
            if invalid_in_a_row > settings.max_tool_retries:
                return _finish(
                    state,
                    run,
                    status="contract_error",
                    stop_reason="contract_validation_failed",
                    final_response="The model returned an invalid decision and bounded repair attempts failed.",
                )
            repair_note = (
                f"Your previous decision was rejected: {detail}\n"
                f"Permitted actions now: {state.allowed_actions()}. Return one corrected JSON decision."
            )
            continue

        invalid_in_a_row = 0
        repair_note = None
        run.events.append(
            {
                "step": step,
                "event": "decision_accepted",
                "status": decision.status,
                "action": decision.action,
                "reasoning": decision.reasoning_summary,
            }
        )
        log.info("decision_accepted step=%s status=%s action=%s", step, decision.status, decision.action)

        if decision.status != "continue":
            return _terminal_from_decision(state, decision, run)

        assert decision.action is not None
        state.selected_action = decision.action
        arguments = _tool_arguments(state, decision.action, decision.arguments)
        tool_result, tool_trace, tool_error = await _execute_tool(state, decision.action, arguments, step, fault)
        state.tool_calls.append(tool_trace)
        run.events.append(
            {
                "step": step,
                "event": "tool_result",
                "tool": decision.action,
                "outcome": tool_trace.outcome,
                "observation": _safe_observation(tool_result),
            }
        )
        log.info("tool_result step=%s tool=%s outcome=%s", step, decision.action, tool_trace.outcome)

        if tool_trace.outcome != "success":
            run.errors.append(tool_error or {"step": step, "type": "tool_error", "tool": decision.action})
            retries_used = 0
            while (
                tool_trace.outcome != "success"
                and retries_used < settings.max_tool_retries
                and state.step_count < state.max_steps
            ):
                retries_used += 1
                state.step_count += 1
                step = state.step_count
                tool_result, tool_trace, tool_error = await _execute_tool(
                    state, decision.action, arguments, step, "none", attempt=retries_used + 1
                )
                state.tool_calls.append(tool_trace)
                run.events.append(
                    {
                        "step": step,
                        "event": "tool_retry",
                        "tool": decision.action,
                        "attempt": retries_used + 1,
                        "outcome": tool_trace.outcome,
                    }
                )
                log.info("tool_retry step=%s tool=%s outcome=%s", step, decision.action, tool_trace.outcome)
                if tool_trace.outcome != "success":
                    run.errors.append(tool_error or {"step": step, "type": "tool_error", "tool": decision.action})

            if tool_trace.outcome != "success":
                return _finish(
                    state,
                    run,
                    status="tool_error",
                    stop_reason="tool_failure",
                    final_response=(
                        f"The tool '{decision.action}' failed repeatedly. "
                        "I stopped without crashing so you can retry with adjusted inputs."
                    ),
                )

        _apply_observation(state, decision.action, tool_result)
        state.last_observation = {"tool": decision.action, "result": _safe_observation(tool_result)}

        # The system, not the model, decides when a stop condition has been met.
        if decision.action == "validate_study_input" and state.validation_issues:
            return _finish(
                state,
                run,
                status="needs_clarification",
                stop_reason="needs_clarification",
                final_response=_clarification_message(state.validation_issues),
            )
        if decision.action == "summarize_plan" and state.summary:
            return _finish(
                state, run, status="completed", stop_reason="goal_completed", final_response=state.summary
            )


def _source_text(request: ArenaRequest, history: list[BaseMessage]) -> str:
    """Text that may ground a task's details: what the user wrote plus supplied notes, never agent replies."""
    parts = [str(m.content) for m in history if isinstance(m, HumanMessage)]
    parts.append(request.task)
    parts.extend(item.content for item in request.external_context)
    return "\n".join(parts)


def _forced_decision(state: AgentState) -> AgentDecision | None:
    """With the model down, continue only if validated state leaves exactly one possible next step."""
    allowed = state.allowed_actions()
    if not state.validated or len(allowed) != 1:
        return None
    return AgentDecision(
        status="continue",
        action=allowed[0],
        reasoning_summary="Model unavailable; the only permitted next step needs no judgement.",
    )


def _terminal_from_decision(state: AgentState, decision: AgentDecision, run: RunLog) -> ArenaResponse:
    mapping = {
        "needs_clarification": ("needs_clarification", "needs_clarification"),
        "blocked": ("blocked", "blocked_action"),
        "failed": ("failed", "failed"),
    }
    status, stop_reason = mapping[decision.status]
    message = decision.user_message or {
        "needs_clarification": "I need more information to continue.",
        "blocked": "This request is outside the agent's permitted autonomy.",
        "failed": "The agent could not continue.",
    }[decision.status]
    run.events.append({"step": state.step_count, "event": "agent_stop", "reason": stop_reason})
    return _finish(state, run, status=status, stop_reason=stop_reason, final_response=message)


def _clarification_message(issues: list[str]) -> str:
    bullets = "; ".join(issues[:4])
    return (
        "I need a bit more information before I can build a reliable study plan. "
        f"Please provide: {bullets} "
        "Include task names, deadlines (day/date), estimated hours, and class times if relevant."
    )


def _finish(state: AgentState, run: RunLog, status: str, stop_reason: str, final_response: str) -> ArenaResponse:
    text = (final_response or "").strip() or "No response."
    if len(text) > 2000:
        text = text[:1960] + "\n[Response shortened to fit the limit.]"
    log.info(
        "run_end request_id=%s status=%s steps=%s stop_reason=%s model_calls=%s tokens_in=%s tokens_out=%s",
        state.request_id,
        status,
        state.step_count,
        stop_reason,
        state.model_calls,
        run.input_tokens,
        run.output_tokens,
    )
    return ArenaResponse(
        request_id=state.request_id,
        status=status,  # type: ignore[arg-type]
        final_response=text,
        steps=state.step_count,
        stop_reason=stop_reason,
        tool_calls=state.tool_calls,
        errors=run.errors,
        events=run.events,
        metrics=Metrics(
            model_calls=state.model_calls,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            estimated_cost_usd=run.cost,
        ),
    )


def _semantic_validate(decision: AgentDecision, state: AgentState) -> None:
    if decision.status == "completed":
        raise SemanticValidationError(
            "status 'completed' is not permitted; the application completes the run after summarize_plan"
        )
    if decision.status != "continue":
        return
    allowed = state.allowed_actions()
    if decision.action not in allowed:
        raise SemanticValidationError(f"Action '{decision.action}' is not permitted now; allowed: {allowed}")
    if decision.action == "validate_study_input":
        for key in ("tasks", "classes", "unavailable"):
            items = decision.arguments.get(key)
            if items is not None and not (isinstance(items, list) and all(isinstance(i, dict) for i in items)):
                raise SemanticValidationError(f"arguments.{key} must be a list of objects")
        window = decision.arguments.get("preferred_hours")
        if window is not None and not isinstance(window, dict):
            raise SemanticValidationError("arguments.preferred_hours must be an object or null")


def _tool_arguments(state: AgentState, action: str, proposed: dict[str, Any]) -> dict[str, Any]:
    """The model only supplies extracted user facts; every later tool reads trusted state."""
    if action == "validate_study_input":
        return _ground_inputs(proposed, state.source_text)
    classes = [c.model_dump() for c in state.classes]
    unavailable = [u.model_dump() for u in state.unavailable]
    if action == "generate_study_schedule":
        return {
            "tasks": [t.model_dump() for t in state.tasks],
            "classes": classes,
            "unavailable": unavailable,
            "preferred_hours": state.preferred_hours,
        }
    schedule = [s.model_dump() for s in state.schedule]
    if action == "check_conflicts":
        return {"schedule": schedule, "classes": classes, "unavailable": unavailable}
    return {
        "schedule": schedule,
        "warnings": state.schedule_warnings,
        "tasks": [t.model_dump() for t in state.tasks],
        "conflicts": state.conflicts,
    }


def _ground_inputs(proposed: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Drop any deadline or duration the model cannot trace to the user's words, so it is asked for."""
    source_tokens = set(re.findall(r"\w+", source_text.lower()))
    grounded = dict(proposed)
    tasks = []
    for raw in proposed.get("tasks") or []:
        task = dict(raw)
        deadline_evidence = task.pop("deadline_evidence", None)
        hours_evidence = task.pop("hours_evidence", None)
        if not _is_grounded(deadline_evidence, source_tokens, SPECIFIC_DATE):
            task["deadline"] = None
        elif task.get("deadline"):
            task["deadline"] = _align_weekday(task["deadline"], str(deadline_evidence))
        if not _is_grounded(hours_evidence, source_tokens, SPECIFIC_HOURS):
            task["estimated_hours"] = None
        tasks.append(task)
    grounded["tasks"] = tasks
    return grounded


def _is_grounded(evidence: Any, source_tokens: set[str], specific: re.Pattern[str]) -> bool:
    if not isinstance(evidence, str) or not specific.search(evidence):
        return False
    return set(re.findall(r"\w+", evidence.lower())) <= source_tokens


def _align_weekday(deadline: Any, evidence: str) -> Any:
    """Models misdate weekdays; when the quote names one, resolve it to its next real date."""
    named = [day for day in WEEKDAYS if re.search(rf"\b{day[:3]}", evidence, re.IGNORECASE)]
    try:
        parsed = datetime.fromisoformat(str(deadline))
    except ValueError:
        return deadline
    if len(named) != 1 or parsed.strftime("%A") == named[0]:
        return deadline
    today = current_time().date()
    offset = (WEEKDAYS.index(named[0]) - today.weekday()) % 7
    return (today + timedelta(days=offset)).isoformat()


async def _execute_tool(
    state: AgentState,
    action: str,
    arguments: dict[str, Any],
    step: int,
    fault: str,
    attempt: int = 1,
) -> tuple[dict[str, Any], ToolTrace, dict[str, Any] | None]:
    started = perf_counter()
    tool = TOOLS[action]

    def trace(outcome: str) -> ToolTrace:
        latency = (perf_counter() - started) * 1000
        return ToolTrace(step=step, tool=action, attempt=attempt, outcome=outcome, latency_ms=latency)  # type: ignore[arg-type]

    # Inject controlled Arena faults at the tool boundary once per run.
    if fault == "tool_timeout" and not state.fault_triggered:
        state.fault_triggered = True
        await asyncio.sleep(0)  # keep async path realistic without long waits in tests
        return {}, trace("timeout"), {"step": step, "type": "timeout", "tool": action, "detail": "Injected tool timeout"}

    if fault == "malformed_tool_output" and not state.fault_triggered:
        state.fault_triggered = True
        return {"unexpected": True}, trace("malformed_output"), {
            "step": step,
            "type": "malformed_output",
            "tool": action,
            "detail": "Injected malformed tool output",
        }

    try:
        result = await asyncio.wait_for(asyncio.to_thread(tool, arguments), timeout=settings.tool_timeout_seconds)
    except TimeoutError:
        return {}, trace("timeout"), {"step": step, "type": "timeout", "tool": action, "detail": "Tool timed out"}
    except Exception as exc:  # noqa: BLE001
        log.exception("tool_exception tool=%s", action)
        return {}, trace("exception"), {"step": step, "type": "exception", "tool": action, "detail": str(exc)[:300]}

    if not isinstance(result, dict) or "message" not in result:
        return {}, trace("malformed_output"), {
            "step": step,
            "type": "malformed_output",
            "tool": action,
            "detail": "Tool returned unexpected shape",
        }
    return result, trace("success"), None


def _apply_observation(state: AgentState, action: str, result: dict[str, Any]) -> None:
    if action == "validate_study_input":
        state.validated = True
        state.validation_issues = list(result.get("issues") or [])
        state.tasks = [StudyTask.model_validate(t) for t in result.get("tasks") or []]
        state.classes = [ClassSlot.model_validate(c) for c in result.get("classes") or []]
        state.unavailable = [UnavailableSlot.model_validate(u) for u in result.get("unavailable") or []]
        state.preferred_hours = result.get("preferred_hours") or state.preferred_hours
    elif action == "generate_study_schedule":
        state.schedule_generated = True
        state.conflicts_checked = False
        state.schedule = [StudySession.model_validate(s) for s in result.get("schedule") or []]
        state.schedule_warnings = list(result.get("warnings") or [])
        state.conflicts = list(result.get("conflicts") or [])
    elif action == "check_conflicts":
        state.conflicts = list(result.get("conflicts") or [])
        state.conflicts_checked = True
    elif action == "summarize_plan":
        state.summary = result.get("summary") or result.get("message")
        state.final_response = state.summary


def _safe_observation(result: dict[str, Any]) -> dict[str, Any]:
    # Keep events compact for UI/Arena response bounds.
    compact = {k: result.get(k) for k in ("ok", "complete", "message", "conflict_count", "issues", "warnings") if k in result}
    if "schedule" in result:
        compact["schedule_count"] = len(result.get("schedule") or [])
    if "unscheduled_hours" in result:
        compact["unscheduled_hours"] = result["unscheduled_hours"]
    if "summary" in result:
        compact["summary_preview"] = result["summary"][:300]
    if "conflicts" in result:
        compact["conflicts"] = result.get("conflicts")
    return compact
