"""Model provider layer: Groq free tier (live) with model failover; mock only when explicitly selected."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import settings
from app.models import AgentDecision, AgentState
from app.tools import _day_from_deadline, extract_study_context

log = logging.getLogger("studyplan.llm")

MOCK_MODEL = "studyplan-mock-v1"

# Published Groq on-demand prices (USD per 1M input/output tokens). The free tier bills $0;
# these give a paid-equivalent estimate. Models without a public price report null, never 0.
MODEL_PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

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
        model: str | None = None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.estimated_cost_usd = cost
        self.model = model


class ModelUnavailableError(RuntimeError):
    """No configured model could be reached (rate limits, outage, bad key, not configured)."""


class InvalidModelOutputError(ValueError):
    """The model answered, but not with parseable JSON. Handled by the contract-repair path."""

    def __init__(self, detail: str, usage: ModelUsage | None = None):
        super().__init__(detail)
        self.usage = usage or ModelUsage()


class OutputBudgetExceededError(RuntimeError):
    """The model hit MAX_OUTPUT_TOKENS, so its decision was cut off."""

    def __init__(self, detail: str, usage: ModelUsage | None = None):
        super().__init__(detail)
        self.usage = usage or ModelUsage()


class _RateLimitedError(RuntimeError):
    def __init__(self, detail: str, retry_after: float | None):
        super().__init__(detail)
        self.retry_after = retry_after


def is_mock_provider() -> bool:
    return (settings.model_provider or "").strip().lower() == "mock"


def live_credentials_available() -> bool:
    return not is_mock_provider() and bool(settings.groq_api_key.strip())


def configured_models() -> list[str]:
    """Models shown in the UI / accepted by /chat. Never includes secrets."""
    if is_mock_provider():
        return [MOCK_MODEL]
    if not live_credentials_available():
        return []
    extras = [m.strip() for m in settings.fallback_models.split(",") if m.strip()]
    return list(dict.fromkeys([settings.model_name.strip(), *extras]))


def default_run_model() -> str:
    """Model used by POST /arena/run."""
    return MOCK_MODEL if is_mock_provider() else settings.model_name.strip()


async def decide(
    state: AgentState,
    messages: list[BaseMessage],
    model_name: str,
    force_invalid: bool = False,
) -> tuple[dict[str, Any], ModelUsage]:
    """Return raw decision payload + usage. Validation happens in the agent loop."""
    if force_invalid:
        return {"status": "continue", "action": "not_a_real_tool", "arguments": "bad"}, ModelUsage()
    if is_mock_provider():
        return mock_decide(state, messages).model_dump(), ModelUsage(model=MOCK_MODEL)
    return await _live_decide(messages, model_name)


async def _live_decide(messages: list[BaseMessage], model_name: str) -> tuple[dict[str, Any], ModelUsage]:
    """Try the selected model, then the others. Wait out one short rate-limit window, then give up."""
    if not live_credentials_available():
        raise ModelUnavailableError("GROQ_API_KEY is not configured")

    candidates = list(dict.fromkeys([model_name, *configured_models()]))
    failures: list[str] = []
    for attempt in range(2):
        retry_after: list[float] = []
        for model in candidates:
            try:
                return await _call_model(messages, model)
            except _RateLimitedError as exc:
                log.warning("model_rate_limited model=%s retry_after=%s", model, exc.retry_after)
                failures.append(str(exc))
                if exc.retry_after is not None:
                    retry_after.append(exc.retry_after)
            except (RuntimeError, httpx.HTTPError) as exc:
                if isinstance(exc, OutputBudgetExceededError):
                    raise
                log.warning("model_call_failed model=%s error=%s", model, type(exc).__name__)
                failures.append(f"{model}: {type(exc).__name__}: {exc}")
        wait = min(retry_after, default=None)
        if attempt or wait is None or wait > settings.max_rate_limit_wait_seconds:
            break
        await asyncio.sleep(wait)
    raise ModelUnavailableError("; ".join(failures[-3:])[:600])


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=settings.llm_timeout_seconds)


async def _call_model(messages: list[BaseMessage], model: str) -> tuple[dict[str, Any], ModelUsage]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": settings.max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": _role_for(m), "content": str(m.content)} for m in messages],
    }
    if model.startswith("openai/gpt-oss"):
        # Reasoning tokens count against max_tokens, so keep them small and out of the reply.
        payload["reasoning_effort"] = "low"
        payload["include_reasoning"] = False

    async with _client() as client:
        response = await client.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json=payload,
        )

    if response.status_code == 429:
        raise _RateLimitedError(f"{model}: rate limited", _retry_after(response))
    if response.status_code == 400 and "json_validate_failed" in response.text:
        # Groq JSON mode rejects its own invalid generation; the model did answer, so repair it.
        raise InvalidModelOutputError("Model output was not valid JSON.", ModelUsage(model=model))
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {_scrub(response.text)[:200]}")

    data = response.json()
    usage = _usage(data, model)
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected provider response shape") from exc

    if choice.get("finish_reason") == "length":
        raise OutputBudgetExceededError(f"Output cut off at {settings.max_output_tokens} tokens.", usage)
    try:
        decision = _extract_json(content or "")
    except ValueError as exc:
        raise InvalidModelOutputError(f"Model output was not valid JSON: {exc}", usage) from exc
    if not isinstance(decision, dict):
        raise InvalidModelOutputError("Model output must be a JSON object.", usage)
    return decision, usage


def _usage(data: dict[str, Any], model: str) -> ModelUsage:
    usage = data.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    cost = None
    price = MODEL_PRICES.get(model)
    if price and input_tokens is not None and output_tokens is not None:
        cost = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
    return ModelUsage(input_tokens, output_tokens, cost, model)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


def _scrub(text: str) -> str:
    key = settings.groq_api_key.strip()
    return text.replace(key, "<redacted>") if key else text


def _extract_json(text: str) -> Any:
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


def mock_decide(state: AgentState, messages: list[BaseMessage]) -> AgentDecision:
    """Deterministic offline policy for tests. Runs only when MODEL_PROVIDER=mock."""
    history_text = _history_text(messages)

    if any(re.search(pattern, state.goal.lower()) for pattern in BLOCKED_PATTERNS):
        return AgentDecision(
            status="blocked",
            user_message=(
                "I can create a sandbox study plan, but I cannot submit coursework, "
                "send emails, or modify real systems. Please ask only for planning help."
            ),
            reasoning_summary="Autonomy boundary triggered.",
        )

    allowed = state.allowed_actions()
    if not allowed:
        return AgentDecision(
            status="needs_clarification",
            user_message="None of the requested hours fit. Please share more available time.",
            reasoning_summary="Nothing could be scheduled.",
        )

    action = allowed[-1]
    arguments: dict[str, Any] = {}
    if action == "validate_study_input":
        extracted = extract_study_context(state.goal, history_text)
        arguments = {
            "tasks": [
                {
                    **task,
                    "deadline_evidence": _day_from_deadline(task.get("deadline")),
                    "hours_evidence": f"{task['estimated_hours']:g}" if task.get("estimated_hours") else None,
                }
                for task in extracted["tasks"]
            ],
            "classes": extracted["classes"],
            "unavailable": extracted["unavailable"],
            "preferred_hours": extracted["preferred_hours"],
        }
    return AgentDecision(
        status="continue",
        action=action,
        arguments=arguments,
        reasoning_summary=f"Next permitted step: {action}.",
    )


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
