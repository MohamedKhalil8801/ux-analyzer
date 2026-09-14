"""Platform-neutral contracts for structured model providers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from ux_analyzer.domain.attention import CoarseScent, FullScent, PersonaObservation
from ux_analyzer.domain.interface import ViewportSnapshot


class ModelRole(StrEnum):
    """Independent model responsibilities used by benchmark runs."""

    COARSE_SCENT = "coarse-scent"
    FULL_SCENT = "full-scent"
    COGNITIVE = "cognitive"
    REPORT_ANALYST = "report-analyst"
    REPORT_EVIDENCE_AUDITOR = "report-evidence-auditor"
    REPORT_PATTERN_REVIEWER = "report-pattern-reviewer"
    REPORT_ADJUDICATOR = "report-adjudicator"
    REDESIGN_PROPOSER = "redesign-proposer"
    REDESIGN_CRITIC_MERGER = "redesign-critic-merger"

    # ADR 0003 legacy names remain source-compatible but never serialize.
    UX_ANALYST = REPORT_ANALYST
    EVIDENCE_AUDITOR = REPORT_EVIDENCE_AUDITOR
    PATTERN_REVIEWER = REPORT_PATTERN_REVIEWER


class ModelResponseValidationError(ValueError):
    """Sanitized role-output validation failure safe for run evidence."""

    def __init__(
        self,
        role: ModelRole,
        reason: str,
        *,
        response_summary: Mapping[str, Any] | None = None,
    ) -> None:
        self.role = ModelRole(role)
        self.reason = reason
        self.response_summary = dict(response_summary or {})
        super().__init__(f"{self.role.value}: {reason}")


@dataclass(frozen=True, slots=True)
class CognitiveRunContext:
    """Safe run state supplied to the cognitive role between decisions."""

    viewport_id: str
    previous_action: Mapping[str, Any] | None = None
    previous_action_result: Mapping[str, Any] | None = None
    completed_fixture_keys: tuple[str, ...] = ()
    fixture_input_complete: bool = False
    working_memory_capacity: int = 0
    confidence: float = 0.0
    frustration: float = 0.0
    abandonment_threshold: float = 0.0
    attention_temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Provider-neutral chat message."""

    role: str
    content: str
    attachments: tuple[ModelAttachment, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat message role: {self.role!r}")
        if not self.content:
            raise ValueError("chat message content must not be empty")
        attachments = _normalize_model_attachments(self.attachments)
        if len({item.evidence_id for item in attachments}) != len(attachments):
            raise ValueError("chat message attachments must have unique evidence IDs")
        if attachments and self.role != "user":
            raise ValueError("model attachments are supported only on user messages")
        object.__setattr__(self, "attachments", attachments)

    def model_dump(self) -> dict[str, object]:
        return {"role": self.role, "content": self.content}


_SAFE_MODEL_ATTACHMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ModelAttachment:
    """Validated visual evidence available to one model call."""

    evidence_id: str
    path: Path
    media_type: Literal["image/png", "image/jpeg"]
    sha256: str

    def __post_init__(self) -> None:
        if _SAFE_MODEL_ATTACHMENT_ID.fullmatch(self.evidence_id) is None:
            raise ValueError("evidence ID must be non-empty and safe")
        path = Path(self.path)
        if (
            not path.parts
            or not path.name
            or any(part in {".", ".."} or "\x00" in part for part in path.parts)
        ):
            raise ValueError("attachment path must be safe")
        if self.media_type not in {"image/png", "image/jpeg"}:
            raise ValueError("attachment media type is unsupported")
        if _SHA256_DIGEST.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "path", path)


def _normalize_model_attachments(value: object) -> tuple[ModelAttachment, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("chat message attachments must be a sequence")
    sequence = cast(Sequence[object], value)
    normalized: tuple[object, ...] = tuple(sequence)
    if any(not isinstance(item, ModelAttachment) for item in normalized):
        raise TypeError("chat message attachments must be ModelAttachment values")
    return cast(tuple[ModelAttachment, ...], normalized)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry settings for safe, idempotent structured calls."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must not be negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must not be below base delay")
        if self.multiplier < 1:
            raise ValueError("retry multiplier must be at least one")

    def delay_for_retry(self, retry_number: int) -> float:
        """Return bounded exponential delay after one failed attempt."""

        if retry_number < 1:
            raise ValueError("retry number must be at least one")
        return min(
            self.max_delay_seconds,
            self.base_delay_seconds * self.multiplier ** (retry_number - 1),
        )


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported by an OpenAI-compatible response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True, slots=True)
class RetryEvent:
    """Sanitized record of one retry decision."""

    role: ModelRole
    model: str
    attempt: int
    reason: str
    status_code: int | None = None
    delay_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Role-specific provider identity safe to persist in a run bundle."""

    provider_id: str
    role: ModelRole
    model_id: str
    endpoint_origin: str
    prompt_version: str
    schema_version: str
    provider_version: str = "openai-compatible-v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("provider_id", self.provider_id),
            ("model_id", self.model_id),
            ("endpoint_origin", self.endpoint_origin),
            ("prompt_version", self.prompt_version),
            ("schema_version", self.schema_version),
            ("provider_version", self.provider_version),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        object.__setattr__(self, "role", ModelRole(self.role))


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    """Sanitized audit record for one logical structured model call."""

    role: ModelRole
    model: str
    endpoint_origin: str
    prompt_digest: str
    schema_version: str
    attempts: int
    latency_ms: int
    token_usage: TokenUsage
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    retries: tuple[RetryEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("model call needs at least one attempt")
        if self.latency_ms < 0:
            raise ValueError("latency must not be negative")
        object.__setattr__(self, "role", ModelRole(self.role))
        object.__setattr__(self, "retries", tuple(self.retries))


ModelUsageRecord = ModelCallRecord
UsageRecord = ModelCallRecord
RetryRecord = RetryEvent
RoleManifest = ModelManifest


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredModelClient(Protocol):
    """Port consumed by role-specific providers."""

    endpoint_origin: str

    async def complete(
        self,
        schema: type[SchemaT],
        messages: Sequence[ChatMessage],
        model: str,
        role: ModelRole,
    ) -> SchemaT: ...


class CoarseScentEvaluator(Protocol):
    """Port for pre-notice glance-level scent."""

    async def evaluate(
        self, goal: str, snapshot: ViewportSnapshot
    ) -> tuple[CoarseScent, ...]: ...


class FullScentEvaluator(Protocol):
    """Port for post-notice scent."""

    async def evaluate(
        self, goal: str, state: Any, snapshot: ViewportSnapshot
    ) -> tuple[FullScent, ...]: ...


class CognitiveAgent(Protocol):
    """Port for qualitative action selection from persona-visible context."""

    async def decide(self, goal: str, observation: PersonaObservation) -> BaseModel: ...
