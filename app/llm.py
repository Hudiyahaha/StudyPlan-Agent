"""Model provider abstraction with deterministic mock + optional live providers."""
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
    def __init__(self, input_tokens: int | None = None, output_tokens: int | None = None, cost: float | None = None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.estimated_cost_usd = cost


def configured_models() -> list[str]:
    provider = (settings.model_provider or "mock").lower()
    name = settings.model_name or "studyplan-mock-v1"
    models = ["studyplan-mock-v1"]
    if provider == "mock":
        return ["studyplan-mock-v1"]
    if provider in {"openai", "gemini", "anthropic"} and name:
        if name not in models:
            models.append(name)
        return models
    if provider == "unconfigured":
        return ["studyplan-mock-v1"]
    return models


def resolve_provider(model_name: str) -> str:
    if model_name in {"studyplan-mock-v1", "mock", "unconfigured", ""}:
        return "mock"
    provider = (settings.model_provider or "mock").lower()
    if provider in {"openai", "gemini", "anthropic"}:
        return provider
    return "mock"


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
        return decision.model_dump(), ModelUsage(input_tokens=None, output_tokens=None, cost=None)
    try:
        if provider == "openai":
            return await _openai_decide(messages, model_name)
        if provider == "gemini":
            return await _gemini_decide(messages, model_name)
    except Exception as exc:  # noqa: BLE001
        log.warning("Live provider failed (%s); falling back to mock: %s", provider, exc)
        decision = mock_decide(state, messages)
        return decision.model_dump(), ModelUsage()
    log.warning("Unknown provider %s; using mock", provider)
    decision = mock_decide(state, messages)
    return decision.model_dump(), ModelUsage()


def mock_decide(state: AgentState, messages: list[BaseMessage]) -> AgentDecision:
    """Deterministic policy used for local/tests and as fallback."""
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

    # Seed extracted context once if empty.
    if not state.tasks and not state.validated:
        extracted = extract_study_context(combined_goal, history_text)
        # Keep extracted values in arguments for validate tool.
        if not state.last_observation:
            state.tasks = []  # ensure clean
            # Store temporary extraction on state via preferred_hours bag is messy;
            # instead drive the first tool call with arguments.
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

    # If conflicts exist after check, still summarize with warnings rather than looping forever.
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
            # Skip the dynamically assembled giant user block used for live models.
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


async def _openai_decide(messages: list[BaseMessage], model_name: str) -> tuple[dict[str, Any], ModelUsage]:
    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY missing; falling back to mock")
        raise RuntimeError("OPENAI_API_KEY is not configured")
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": settings.max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": _role_for(m), "content": str(m.content)} for m in messages],
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return _extract_json(content), ModelUsage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        cost=None,
    )


async def _gemini_decide(messages: list[BaseMessage], model_name: str) -> tuple[dict[str, Any], ModelUsage]:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    # Convert messages to Gemini contents; system becomes a leading user preamble.
    system_bits = [str(m.content) for m in messages if m.type == "system"]
    contents = []
    if system_bits:
        contents.append({"role": "user", "parts": [{"text": "SYSTEM:\n" + "\n".join(system_bits)}]})
    for message in messages:
        if message.type == "system":
            continue
        role = "user" if message.type == "human" else "model"
        contents.append({"role": role, "parts": [{"text": str(message.content)}]})
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        f"?key={settings.gemini_api_key}"
    )
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": settings.max_output_tokens,
            "responseMimeType": "application/json",
        },
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage_meta = data.get("usageMetadata") or {}
    return _extract_json(text), ModelUsage(
        input_tokens=usage_meta.get("promptTokenCount"),
        output_tokens=usage_meta.get("candidatesTokenCount"),
        cost=None,
    )


def _role_for(message: BaseMessage) -> str:
    if message.type == "system":
        return "system"
    if message.type == "ai":
        return "assistant"
    return "user"


# Silence unused import warning in type checkers for TOOLS documentation linkage
_ = TOOLS
