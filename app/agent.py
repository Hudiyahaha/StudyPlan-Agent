"""Bounded StudyPlan Agent controller. System owns validation and stopping."""
from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import BaseMessage

from app.config import settings
from app.llm import decide
from app.models import (
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


class SemanticValidationError(ValueError):
    """Raised when a decision is schema-valid but semantically illegal."""


async def run_agent(request: ArenaRequest, history: list[BaseMessage] | None, model: str) -> ArenaResponse:
    history = history or []
    state = AgentState(
        request_id=request.request_id,
        goal=request.task.strip(),
        max_steps=min(request.arena_config.max_steps, settings.max_steps),
        preferred_hours={
            "start": settings.preferred_start,
            "end": settings.preferred_end,
        },
    )
    fault = request.arena_config.fault.type
    events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None

    log.info(
        "run_start request_id=%s max_steps=%s fault=%s model=%s",
        request.request_id,
        state.max_steps,
        fault,
        model,
    )

    while True:
        if state.step_count >= state.max_steps:
            return _finish(
                state,
                status="budget_exceeded",
                stop_reason="max_steps_exceeded",
                final_response=(
                    state.final_response
                    or state.summary
                    or "Stopped because the step budget was exhausted before the plan could be finished."
                ),
                events=events,
                errors=errors,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )

        state.step_count += 1
        step = state.step_count
        repair_note = None
        decision: AgentDecision | None = None
        raw_decision: dict[str, Any] | None = None

        # Bounded repair loop for invalid model decisions (counts as same step's repairs;
        # each model call also increments model_calls and counts toward budget via step_count
        # on subsequent outer-loop iterations if we escalate).
        for repair_idx in range(settings.max_tool_retries + 1):
            force_invalid = (
                fault == "invalid_agent_decision"
                and not state.fault_triggered
                and repair_idx == 0
            )
            messages = build_messages(
                state,
                history,
                request.external_context,
                repair_note=repair_note,
            )
            try:
                raw_decision, usage = await decide(
                    state,
                    messages,
                    model_name=model,
                    force_invalid=force_invalid,
                )
                if force_invalid:
                    state.fault_triggered = True
                state.model_calls += 1
                if usage.input_tokens is not None:
                    total_input_tokens = (total_input_tokens or 0) + usage.input_tokens
                if usage.output_tokens is not None:
                    total_output_tokens = (total_output_tokens or 0) + usage.output_tokens
            except Exception as exc:  # noqa: BLE001
                log.exception("model_call_failed")
                errors.append({"step": step, "type": "model_error", "detail": str(exc)})
                events.append({"step": step, "event": "model_error", "detail": str(exc)})
                if repair_idx >= settings.max_tool_retries:
                    return _finish(
                        state,
                        status="failed",
                        stop_reason="model_error",
                        final_response="The model call failed and recovery attempts were exhausted.",
                        events=events,
                        errors=errors,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                    )
                repair_note = f"Previous model call failed: {exc}. Return a valid AgentDecision JSON object."
                state.repair_attempts += 1
                continue

            try:
                decision = AgentDecision.model_validate(raw_decision)
                _semantic_validate(decision, state)
                events.append(
                    {
                        "step": step,
                        "event": "decision_accepted",
                        "status": decision.status,
                        "action": decision.action,
                        "reasoning": decision.reasoning_summary,
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001
                state.repair_attempts += 1
                errors.append(
                    {
                        "step": step,
                        "type": "contract_error",
                        "detail": str(exc),
                        "raw": raw_decision,
                    }
                )
                events.append(
                    {
                        "step": step,
                        "event": "decision_rejected",
                        "detail": str(exc),
                        "repair": repair_idx,
                    }
                )
                log.info("decision_rejected step=%s detail=%s", step, exc)
                if repair_idx >= settings.max_tool_retries:
                    return _finish(
                        state,
                        status="contract_error",
                        stop_reason="contract_validation_failed",
                        final_response=(
                            "The model returned an invalid decision and bounded repair attempts failed."
                        ),
                        events=events,
                        errors=errors,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                    )
                repair_note = (
                    f"Your previous decision was invalid: {exc}. "
                    "Return a corrected AgentDecision JSON object."
                )
                # Invalid decisions burn budget: each repair after the first consumes an extra step.
                if state.step_count >= state.max_steps:
                    return _finish(
                        state,
                        status="budget_exceeded",
                        stop_reason="max_steps_exceeded",
                        final_response="Stopped while repairing invalid decisions: step budget exhausted.",
                        events=events,
                        errors=errors,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                    )
                state.step_count += 1
                step = state.step_count

        assert decision is not None

        if decision.status != "continue":
            return _terminal_from_decision(
                state,
                decision,
                events=events,
                errors=errors,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )

        assert decision.action is not None
        state.selected_action = decision.action
        tool_result, tool_trace, tool_error = await _execute_tool(
            state=state,
            action=decision.action,
            arguments=decision.arguments,
            step=step,
            fault=fault,
        )
        state.tool_calls.append(tool_trace)
        events.append(
            {
                "step": step,
                "event": "tool_result",
                "tool": decision.action,
                "outcome": tool_trace.outcome,
                "observation": _safe_observation(tool_result),
            }
        )

        if tool_trace.outcome != "success":
            errors.append(tool_error or {"step": step, "type": "tool_error", "tool": decision.action})
            # Retry within policy using additional steps.
            retries_used = 0
            recovered = False
            while retries_used < settings.max_tool_retries and state.step_count < state.max_steps:
                retries_used += 1
                state.step_count += 1
                step = state.step_count
                tool_result, tool_trace, tool_error = await _execute_tool(
                    state=state,
                    action=decision.action,
                    arguments=decision.arguments,
                    step=step,
                    fault="none",  # fault triggers only once
                    attempt=retries_used + 1,
                )
                state.tool_calls.append(tool_trace)
                events.append(
                    {
                        "step": step,
                        "event": "tool_retry",
                        "tool": decision.action,
                        "attempt": retries_used + 1,
                        "outcome": tool_trace.outcome,
                    }
                )
                if tool_trace.outcome == "success":
                    recovered = True
                    break
                errors.append(tool_error or {"step": step, "type": "tool_error", "tool": decision.action})

            if not recovered:
                return _finish(
                    state,
                    status="tool_error",
                    stop_reason="tool_failure",
                    final_response=(
                        f"The tool '{decision.action}' failed repeatedly. "
                        "I stopped without crashing so you can retry with adjusted inputs."
                    ),
                    events=events,
                    errors=errors,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )

        _apply_observation(state, decision.action, tool_result)
        state.last_observation = {
            "tool": decision.action,
            "result": _safe_observation(tool_result),
        }


def _terminal_from_decision(
    state: AgentState,
    decision: AgentDecision,
    events: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    input_tokens: int | None,
    output_tokens: int | None,
) -> ArenaResponse:
    mapping = {
        "needs_clarification": ("needs_clarification", "needs_clarification"),
        "completed": ("completed", "goal_completed"),
        "blocked": ("blocked", "blocked_action"),
        "failed": ("failed", "failed"),
    }
    status, stop_reason = mapping[decision.status]
    message = decision.user_message or {
        "needs_clarification": "I need more information to continue.",
        "completed": state.summary or "Study plan completed.",
        "blocked": "This request is outside the agent's permitted autonomy.",
        "failed": "The agent could not continue.",
    }[decision.status]
    events.append({"step": state.step_count, "event": "agent_stop", "reason": stop_reason})
    return _finish(
        state,
        status=status,
        stop_reason=stop_reason,
        final_response=message,
        events=events,
        errors=errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _finish(
    state: AgentState,
    status: str,
    stop_reason: str,
    final_response: str,
    events: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    input_tokens: int | None,
    output_tokens: int | None,
) -> ArenaResponse:
    text = (final_response or "No response.").strip()
    if len(text) > 2000:
        text = text[:1990] + "…"
    if len(text) < 1:
        text = "No response."
    log.info(
        "run_end request_id=%s status=%s steps=%s stop_reason=%s",
        state.request_id,
        status,
        state.step_count,
        stop_reason,
    )
    return ArenaResponse(
        request_id=state.request_id,
        status=status,  # type: ignore[arg-type]
        final_response=text,
        steps=min(state.step_count, 6),
        stop_reason=stop_reason,
        tool_calls=state.tool_calls,
        errors=errors,
        events=events,
        metrics=Metrics(
            model_calls=min(state.model_calls, 6),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=None,
        ),
    )


def _semantic_validate(decision: AgentDecision, state: AgentState) -> None:
    if decision.status == "continue":
        if decision.action not in TOOLS:
            raise SemanticValidationError(f"Unknown tool action: {decision.action}")
        if not isinstance(decision.arguments, dict):
            raise SemanticValidationError("arguments must be an object")
        # State-gated actions
        if decision.action == "generate_study_schedule" and state.validation_issues:
            raise SemanticValidationError("Cannot generate a schedule while validation issues remain")
        if decision.action == "check_conflicts":
            schedule = decision.arguments.get("schedule") or [s.model_dump() for s in state.schedule]
            if not schedule:
                raise SemanticValidationError("check_conflicts requires a schedule")
        if decision.action == "summarize_plan":
            schedule = decision.arguments.get("schedule") or [s.model_dump() for s in state.schedule]
            if not schedule and not state.schedule:
                raise SemanticValidationError("summarize_plan requires a schedule")
        # Reject clearly impossible durations in arguments.tasks if present
        for task in decision.arguments.get("tasks") or []:
            hours = task.get("estimated_hours")
            if hours is not None and hours <= 0:
                raise SemanticValidationError("estimated_hours must be positive")
            if hours is not None and hours > 40:
                raise SemanticValidationError("estimated_hours is unrealistically large")
        for session in decision.arguments.get("schedule") or []:
            if session.get("end") and session.get("start") and session["end"] <= session["start"]:
                raise SemanticValidationError("session end_time must be after start_time")
    elif decision.action is not None:
        raise SemanticValidationError("terminal decisions cannot include a tool action")


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

    # Inject controlled Arena faults at the tool boundary once per run.
    if fault == "tool_timeout" and not state.fault_triggered:
        state.fault_triggered = True
        await asyncio.sleep(0)  # keep async path realistic without long waits in tests
        latency = (perf_counter() - started) * 1000
        trace = ToolTrace(
            step=min(step, 6),
            tool=action,
            attempt=attempt,
            outcome="timeout",
            latency_ms=latency,
        )
        return {}, trace, {"step": step, "type": "timeout", "tool": action, "detail": "Injected tool timeout"}

    if fault == "malformed_tool_output" and not state.fault_triggered:
        state.fault_triggered = True
        latency = (perf_counter() - started) * 1000
        trace = ToolTrace(
            step=min(step, 6),
            tool=action,
            attempt=attempt,
            outcome="malformed_output",
            latency_ms=latency,
        )
        return {"unexpected": True}, trace, {
            "step": step,
            "type": "malformed_output",
            "tool": action,
            "detail": "Injected malformed tool output",
        }

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(tool, arguments),
            timeout=settings.tool_timeout_seconds,
        )
        if not isinstance(result, dict) or "message" not in result:
            latency = (perf_counter() - started) * 1000
            trace = ToolTrace(
                step=min(step, 6),
                tool=action,
                attempt=attempt,
                outcome="malformed_output",
                latency_ms=latency,
            )
            return {}, trace, {
                "step": step,
                "type": "malformed_output",
                "tool": action,
                "detail": "Tool returned unexpected shape",
            }
        latency = (perf_counter() - started) * 1000
        trace = ToolTrace(
            step=min(step, 6),
            tool=action,
            attempt=attempt,
            outcome="success",
            latency_ms=latency,
        )
        return result, trace, None
    except TimeoutError:
        latency = (perf_counter() - started) * 1000
        trace = ToolTrace(
            step=min(step, 6),
            tool=action,
            attempt=attempt,
            outcome="timeout",
            latency_ms=latency,
        )
        return {}, trace, {"step": step, "type": "timeout", "tool": action, "detail": "Tool timed out"}
    except Exception as exc:  # noqa: BLE001
        log.exception("tool_exception tool=%s", action)
        latency = (perf_counter() - started) * 1000
        trace = ToolTrace(
            step=min(step, 6),
            tool=action,
            attempt=attempt,
            outcome="exception",
            latency_ms=latency,
        )
        return {}, trace, {"step": step, "type": "exception", "tool": action, "detail": str(exc)}


def _apply_observation(state: AgentState, action: str, result: dict[str, Any]) -> None:
    if action == "validate_study_input":
        state.validated = True
        state.validation_issues = list(result.get("issues") or [])
        state.tasks = [StudyTask.model_validate(t) for t in result.get("tasks") or []]
        state.classes = [ClassSlot.model_validate(c) for c in result.get("classes") or []]
        state.unavailable = [UnavailableSlot.model_validate(u) for u in result.get("unavailable") or []]
        state.preferred_hours = result.get("preferred_hours") or state.preferred_hours
    elif action == "generate_study_schedule":
        state.schedule = [StudySession.model_validate(s) for s in result.get("schedule") or []]
        state.schedule_warnings = list(result.get("warnings") or [])
        state.conflicts = list(result.get("conflicts") or [])
    elif action == "check_conflicts":
        state.conflicts = list(result.get("conflicts") or [])
        state.conflicts_checked = True
        if state.conflicts:
            state.schedule_warnings = list(
                dict.fromkeys(state.schedule_warnings + [f"Found {len(state.conflicts)} conflict(s)."])
            )
    elif action == "summarize_plan":
        state.summary = result.get("summary") or result.get("message")
        state.final_response = state.summary


def _safe_observation(result: dict[str, Any]) -> dict[str, Any]:
    # Keep events compact for UI/Arena response bounds.
    compact = {k: result.get(k) for k in ("ok", "complete", "message", "conflict_count", "issues", "warnings") if k in result}
    if "schedule" in result:
        compact["schedule_count"] = len(result.get("schedule") or [])
    if "summary" in result:
        summary = result["summary"]
        compact["summary_preview"] = summary[:300]
    if "conflicts" in result:
        compact["conflicts"] = result.get("conflicts")
    return compact
