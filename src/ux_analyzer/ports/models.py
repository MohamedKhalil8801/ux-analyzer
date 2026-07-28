"""Platform-neutral contracts for structured model providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from ux_analyzer.domain.attention import CoarseScent, FullScent, ProgressiveObservation
from ux_analyzer.domain.interface import ViewportSnapshot


class ModelRole(StrEnum):
    """Independent model responsibilities used by benchmark runs."""

    COARSE_SCENT = "coarse-scent"
    FULL_SCENT = "full-scent"
    COGNITIVE = "cognitive"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Provider-neutral chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat message role: {self.role!r}")
        if not self.content:
            raise ValueError("chat message content must not be empty")

    def model_dump(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


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

    async def decide(
        self, goal: str, observation: ProgressiveObservation
    ) -> BaseModel: ...
