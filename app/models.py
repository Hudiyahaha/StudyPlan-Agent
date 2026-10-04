"""Arena HTTP contracts plus StudyPlan Agent domain models."""
from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExternalContext(Contract):
    source: str = Field(min_length=1, max_length=100)
    content: str = Field(max_length=10000)
    trust: Literal["untrusted"] = "untrusted"


class Fault(Contract):
    type: Literal[
        "none",
        "tool_timeout",
        "malformed_tool_output",
        "invalid_agent_decision",
    ] = "none"
    trigger: Literal["first_matching_operation"] = "first_matching_operation"


class ArenaConfig(Contract):
    max_steps: int = Field(default=6, ge=1, le=6)
    fault: Fault = Field(default_factory=Fault)

    @field_validator("fault", mode="before")
    @classmethod
    def accept_assignment_shorthand(cls, value: Any) -> Any:
        return {"type": value} if isinstance(value, str) else value


class ArenaRequest(Contract):
    arena_version: Literal["0.1"] = "0.1"
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=80)
    task: str = Field(min_length=1, max_length=10000)
    external_context: list[ExternalContext] = Field(default_factory=list, max_length=20)
    arena_config: ArenaConfig = Field(default_factory=ArenaConfig)

    @field_validator("task")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Task must not be blank")
        return value


class ToolTrace(Contract):
    step: int = Field(ge=1, le=6)
    tool: str
    attempt: int = Field(default=1, ge=1, le=3)
    outcome: Literal["success", "timeout", "malformed_output", "rejected", "exception"]
    latency_ms: float = Field(default=0, ge=0)


class Metrics(Contract):
    latency_ms: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0, le=6)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class ArenaResponse(Contract):
    arena_version: Literal["0.1"] = "0.1"
    request_id: str
    status: Literal[
        "completed",
        "needs_clarification",
        "blocked",
        "approval_required",
        "tool_error",
        "contract_error",
        "budget_exceeded",
        "failed",
    ]
    final_response: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=0, ge=0, le=6)
    stop_reason: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)


class ChatRequest(ArenaRequest):
    session_id: str = Field(min_length=16, max_length=80)
    model: str = Field(default="unconfigured", max_length=120)


# --- Domain models ---------------------------------------------------------

Priority = Literal["low", "medium", "high"]
ToolName = Literal[
    "validate_study_input",
    "check_conflicts",
    "generate_study_schedule",
    "summarize_plan",
]
DecisionStatus = Literal[
    "continue",
    "needs_clarification",
    "completed",
    "blocked",
    "failed",
]


class StudyTask(Contract):
    name: str = Field(min_length=1, max_length=200)
    deadline: str | None = None
    estimated_hours: float | None = Field(default=None, ge=0, le=40)
    priority: Priority = "medium"
    notes: str = ""

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Task name must not be blank")
        return value


class ClassSlot(Contract):
    day: str = Field(min_length=1, max_length=20)
    start: str = Field(min_length=4, max_length=5)
    end: str = Field(min_length=4, max_length=5)
    title: str = "Class"

    @model_validator(mode="after")
    def validate_range(self) -> ClassSlot:
        if self.end <= self.start:
            raise ValueError("Class end must be after start")
        return self


class UnavailableSlot(Contract):
    day: str = Field(min_length=1, max_length=20)
    start: str = Field(min_length=4, max_length=5)
    end: str = Field(min_length=4, max_length=5)
    reason: str = "unavailable"

    @model_validator(mode="after")
    def validate_range(self) -> UnavailableSlot:
        if self.end <= self.start:
            raise ValueError("Unavailable end must be after start")
        return self


class StudySession(Contract):
    task_name: str
    day: str
    start: str
    end: str
    duration_hours: float = Field(gt=0, le=8)
    date: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> StudySession:
        if self.end <= self.start:
            raise ValueError("Session end must be after start")
        return self


class AgentDecision(Contract):
    status: DecisionStatus
    action: ToolName | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = None
    reasoning_summary: str | None = None

    @model_validator(mode="after")
    def validate_decision_shape(self) -> AgentDecision:
        if self.status == "continue" and not self.action:
            raise ValueError("continue decisions require an action")
        if self.status != "continue" and self.action is not None:
            # Allow action only while continuing; terminal statuses clear tools.
            raise ValueError("terminal statuses must not select a tool action")
        if self.arguments is None:
            raise ValueError("arguments must be a dict")
        return self


class AgentState(BaseModel):
    """Per-run execution state owned by the agent controller."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    goal: str
    step_count: int = 0
    max_steps: int = 6
    tasks: list[StudyTask] = Field(default_factory=list)
    classes: list[ClassSlot] = Field(default_factory=list)
    unavailable: list[UnavailableSlot] = Field(default_factory=list)
    preferred_hours: dict[str, Any] = Field(default_factory=dict)
    validated: bool = False
    validation_issues: list[str] = Field(default_factory=list)
    schedule: list[StudySession] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    conflicts_checked: bool = False
    schedule_warnings: list[str] = Field(default_factory=list)
    summary: str | None = None
    selected_action: str | None = None
    last_observation: dict[str, Any] = Field(default_factory=dict)
    status: str = "running"
    errors: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    final_response: str | None = None
    stop_reason: str | None = None
    model_calls: int = 0
    repair_attempts: int = 0
    fault_triggered: bool = False
    blocked_reason: str | None = None
