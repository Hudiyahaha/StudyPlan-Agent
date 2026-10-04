"""Model provider abstraction: Grok (xAI) for submission, mock only as fallback."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings
from app.models import AgentDecision, AgentState
from app.tools import TOOLS, extract_study_context

log = logging.getLogger("studyplan.llm")

# Two Grok tiers for the required model-selection experiment.
GROK_PRIMARY = "grok-4.3"
GROK_SECONDARY = "grok-4.6"

BLOCKED_PATTERNS = [
    r"\bsubmit\b.+\b(assignment|coursework|homework|quiz|form)\b",
    r"\bsend\b.+\bemail\b",
    r"\bdelete\b.+\bfile",
    r"\bpurchase\b|\bbuy\b",
    r"\bpost\b.+\b(lms|portal|blackboard|classroom)\b",
    r"\bmodify\b.+\b(real\s+)?calendar\b",
    r"\balter\b.+\baccount\b",
]


class ModelUsage:
    def __init__(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: float | None = None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.estimated_cost_usd = cost


def live_credentials_available(provider: str | None = None) -> bool:
    provider = (provider or settings.model_provider or "").lower()
    if provider in {"grok", "xai"}:
        return bool(settings.xai_api_key.strip())
    return False


def configured_models() -> list[str]:
    """Models shown in the UI / accepted by /chat. Never includes secrets."""
    provider = (settings.model_provider or "mock").lower()
    name = (settings.model_name or "").strip()

    if provider in {"grok", "xai"} and live_credentials_available("grok"):
        models = [name or GROK_PRIMARY]
        for extra in (GROK_PRIMARY, GROK_SECONDARY):
            if extra not in models:
                models.append(extra)
        models.append("studyplan-mock-v1")
        return models

    return ["studyplan-mock-v1"]


def resolve_provider(model_name: str) -> str:
    if model_name in {"studyplan-mock-v1", "mock", "unconfigured", ""}:
        return "mock"
    if model_name.startswith("grok"):
        return "grok" if live_credentials_available("grok") else "mock"
    provider = (settings.model_provider or "mock").lower()
    if provider in {"grok", "xai"} and live_credentials_available("grok"):
        return "grok"
    return "mock"


def default_run_model() -> str:
    """Model used by POST /arena/run."""
    provider = (settings.model_provider or "mock").lower()
    if provider in {"grok", "xai"} and live_credentials_available("grok"):
        return settings.model_name.strip() or GROK_PRIMARY
    if provider == "mock":
        return "studyplan-mock-v1"
    log.warning(
        "MODEL_PROVIDER=%s but XAI_API_KEY is missing; using studyplan-mock-v1.",
        provider,
    )
    return "studyplan-mock-v1"


async def decide(
    state: AgentState,
    messages: list[BaseMessage],
    model_name: str,
    force_invalid: bool = False,
) -> tuple[dict[str, Any], ModelUsage]:
    """Return raw decision payload + usage. Validation happens in the agent loop."""
    if force_invalid:
        return {"status": "continue", "action": "not_a_real_tool", "arguments": "bad"}, ModelUsage()

    provider = resolve_provider(model_name)
    if provider == "mock":
        decision = mock_decide(state, messages)
        return decision.model_dump(), ModelUsage()

    try:
        if provider == "grok":
            return await _grok_decide(messages, model_name)
    except Exception as exc:  # noqa: BLE001
        log.warning("Grok call failed; falling back to mock: %s", exc)
        decision = mock_decide(state, messages)
        return decision.model_dump(), ModelUsage()

    decision = mock_decide(state, messages)
    return decision.model_dump(), ModelUsage()


def mock_decide(state: AgentState, messages: list[BaseMessage]) -> AgentDecision:
    """Deterministic offline policy used when no live key is available."""
    history_text = _history_text(messages)
    combined_goal = state.goal
    text_blob = f"{history_text}\n{combined_goal}"

    if _is_blocked_request(text_blob):
        return AgentDecision(
            status="blocked",
            action=None,
            arguments={},
            user_message=(
                "I can create a sandbox study plan, but I cannot submit coursework, "
                "send emails, or modify real systems. Please ask only for planning help."
            ),
            reasoning_summary="Autonomy boundary triggered.",
        )

    if not state.tasks and not state.validated:
        extracted = extract_study_context(combined_goal, history_text)
        return AgentDecision(
            status="continue",
            action="validate_study_input",
            arguments={
                "tasks": extracted["tasks"],
                "classes": extracted["classes"],
                "unavailable": extracted["unavailable"],
                "preferred_hours": extracted["preferred_hours"],
            },
            user_message="Checking whether your study details are complete enough to schedule.",
            reasoning_summary="Initial validation of extracted study context.",
        )

    if not state.validated:
        extracted = extract_study_context(combined_goal, history_text)
        return AgentDecision(
            status="continue",
            action="validate_study_input",
            arguments={
                "tasks": [t.model_dump() for t in state.tasks] or extracted["tasks"],
                "classes": [c.model_dump() for c in state.classes] or extracted["classes"],
                "unavailable": [u.model_dump() for u in state.unavailable] or extracted["unavailable"],
                "preferred_hours": state.preferred_hours or extracted["preferred_hours"],
            },
            user_message="Validating study inputs.",
            reasoning_summary="Inputs not validated yet.",
        )

    if state.validation_issues:
        return AgentDecision(
            status="needs_clarification",
            action=None,
            arguments={},
            user_message=_clarification_message(state.validation_issues),
            reasoning_summary="Missing or invalid study details.",
        )

    if not state.schedule:
        return AgentDecision(
            status="continue",
            action="generate_study_schedule",
            arguments={
                "tasks": [t.model_dump() for t in state.tasks],
                "classes": [c.model_dump() for c in state.classes],
                "unavailable": [u.model_dump() for u in state.unavailable],
                "preferred_hours": state.preferred_hours,
            },
            user_message="Generating a conflict-aware schedule from your constraints.",
            reasoning_summary="Validated inputs; next create schedule.",
        )

    if state.schedule and not state.conflicts_checked:
        return AgentDecision(
            status="continue",
            action="check_conflicts",
            arguments={
                "schedule": [s.model_dump() for s in state.schedule],
                "classes": [c.model_dump() for c in state.classes],
                "unavailable": [u.model_dump() for u in state.unavailable],
            },
            user_message="Checking the draft schedule for overlaps with classes and unavailable times.",
            reasoning_summary="Conflict check before finalizing.",
        )

    if not state.summary:
        return AgentDecision(
            status="continue",
            action="summarize_plan",
            arguments={
                "schedule": [s.model_dump() for s in state.schedule],
                "warnings": state.schedule_warnings,
                "tasks": [t.model_dump() for t in state.tasks],
                "conflicts": state.conflicts,
            },
            user_message="Preparing a readable study plan summary.",
            reasoning_summary="Summarize schedule for the student.",
        )

    return AgentDecision(
        status="completed",
        action=None,
        arguments={},
        user_message=state.summary,
        reasoning_summary="Plan ready.",
    )


def _clarification_message(issues: list[str]) -> str:
    bullets = "; ".join(issues[:4])
    return (
        "I need a bit more information before I can build a reliable study plan. "
        f"Please provide: {bullets}. "
        "Include task names, deadlines (day/date), estimated hours, and class times if relevant."
    )


def _is_blocked_request(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in BLOCKED_PATTERNS)


def _history_text(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            content = str(message.content)
            if content.startswith("USER GOAL / CURRENT REQUEST:"):
                continue
            parts.append(content)
        elif isinstance(message, AIMessage):
            parts.append(str(message.content))
    return "\n".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _role_for(message: BaseMessage) -> str:
    if message.type == "system":
        return "system"
    if message.type == "ai":
        return "assistant"
    return "user"


async def _grok_decide(messages: list[BaseMessage], model_name: str) -> tuple[dict[str, Any], ModelUsage]:
    if not settings.xai_api_key:
        raise RuntimeError("XAI_API_KEY is not configured")

    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": settings.max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": _role_for(m), "content": str(m.content)} for m in messages],
    }
    async with httpx.AsyncClient(timeout=40) as client:
        response = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.xai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        # Some Grok models may reject response_format; retry once without it.
        if response.status_code >= 400 and "response_format" in response.text.lower():
            payload.pop("response_format", None)
            response = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.xai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"Grok HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Grok response shape: {data}") from exc

    usage = data.get("usage") or {}
    return _extract_json(content), ModelUsage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        cost=None,
    )


_ = TOOLS
