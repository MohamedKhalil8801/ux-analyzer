"""OpenAI-compatible structured model adapter.

This module is infrastructure. Domain and application code depend only on the
contracts in ``ux_analyzer.ports.models``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values, load_dotenv
from PIL import Image
from pydantic import BaseModel, ValidationError

from ux_analyzer.ports.model_transport import (
    TransportBudgetError,
    enforce_transport_size,
    serialize_transport_json,
    transport_size,
)
from ux_analyzer.ports.models import (
    ChatMessage,
    ModelAttachment,
    ModelCallRecord,
    ModelManifest,
    ModelRole,
    RetryEvent,
    RetryPolicy,
    StructuredModelClient,
    TokenUsage,
)
from ux_analyzer.storage.run_bundle import (
    secure_assert_ancestors,
    secure_read_bytes,
)

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_REDACTED = "[REDACTED]"
_REDACTED_ATTACHMENT = "[REDACTED_ATTACHMENT]"
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_DATA_URI_PATTERN = re.compile(r"(?i)data:image/(?:png|jpeg);base64,[A-Za-z0-9+/=]+")
_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_REPORT_ROLES = frozenset(
    {
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_EVIDENCE_AUDITOR,
        ModelRole.REPORT_PATTERN_REVIEWER,
        ModelRole.REPORT_ADJUDICATOR,
    }
)
_SAFE_FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "function_call", "content_filter"}
)
_SAFE_TEXT_PART_TYPES = frozenset({"text", "output_text"})
_MAX_MODEL_ATTACHMENT_BYTES = 16 * 1024 * 1024
_MAX_MODEL_RESPONSE_BYTES = 1_000_000
_MAX_DIAGNOSTIC_CONTENT_LENGTH = 1_000_000
_SAFE_STRUCTURAL_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_REPORT_JSON_SCALAR = re.compile(
    r'(?<![A-Za-z0-9_-])(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|"(?:\\.|[^"\\])*")(?![A-Za-z0-9_-])'
)


def _safe_provider_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 256:
        return None
    if re.fullmatch(r"[A-Za-z0-9._:/-]+", value) is None:
        return None
    return value


class ModelConfigurationError(ValueError):
    """Raised when model environment/configuration is incomplete or unsafe."""


class ModelFailureError(RuntimeError):
    """Terminal model failure after classification and bounded retries."""

    def __init__(
        self,
        reason: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
        request_id: str | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code if type(status_code) is int else None
        self.error_code = _safe_provider_text(error_code)
        self.error_type = _safe_provider_text(error_type)
        self.request_id = _safe_provider_text(request_id)
        self.diagnostics = dict(diagnostics or {})


class _ModelResponseTooLarge(RuntimeError):
    """Raised before an oversized provider response is materialized or parsed."""


def load_environment_file(
    dotenv_path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> None:
    """Load local dotenv values without overriding explicit environment values."""

    path = dotenv_path or Path.cwd() / ".env"
    if environ is None:
        load_dotenv(dotenv_path=path, override=False)
        return
    for name, value in dotenv_values(path).items():
        if value is not None:
            environ.setdefault(name, value)


def sanitize_for_log(value: object, *, secrets: Sequence[str] = ()) -> Any:
    """Redact secret keys and exact secret values from nested log payloads."""

    secret_values = tuple(secret for secret in secrets if secret)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): (
                _REDACTED
                if str(key).lower() in _SENSITIVE_KEYS
                else sanitize_for_log(item, secrets=secret_values)
            )
            for key, item in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[object], value)
        return [sanitize_for_log(item, secrets=secret_values) for item in sequence]
    if isinstance(value, str):
        sanitized = _BEARER_PATTERN.sub("Bearer " + _REDACTED, value)
        sanitized = _DATA_URI_PATTERN.sub(_REDACTED_ATTACHMENT, sanitized)
        for secret in sorted(secret_values, key=len, reverse=True):
            sanitized = sanitized.replace(secret, _REDACTED)
        return sanitized
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _REDACTED_ATTACHMENT
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _endpoint_origin(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelConfigurationError("UXA_LLM_BASE_URL must be an HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ModelConfigurationError("UXA_LLM_BASE_URL must not contain credentials")
    return f"{parsed.scheme}://{parsed.netloc}"


def _as_int(value: object, *, name: str) -> int:
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise ModelConfigurationError(f"{name} must be numeric")
    return int(value)


def _as_float(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise ModelConfigurationError(f"{name} must be numeric")
    return float(value)


def _timeout_seconds(value: object, *, mode: Literal["api", "codex"]) -> float | None:
    if value is None:
        if mode != "codex":
            raise ModelConfigurationError(
                "unbounded timeout is supported only in codex mode"
            )
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "none",
        "off",
        "unlimited",
    }:
        if mode != "codex":
            raise ModelConfigurationError(
                "unbounded timeout is supported only in codex mode"
            )
        return None
    parsed = _as_float(value, name="timeout_seconds")
    if parsed <= 0:
        raise ModelConfigurationError("timeout_seconds must be greater than zero")
    return parsed


def _normalize_reasoning_effort(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    normalized_effort = value.strip().lower()
    if normalized_effort not in _REASONING_EFFORTS:
        raise ModelConfigurationError(
            f"{name} must be one of: none, minimal, low, medium, high, xhigh, max"
        )
    return normalized_effort


def _normalize_llm_mode(value: object) -> Literal["api", "codex"]:
    if not isinstance(value, str):
        raise ModelConfigurationError("UXA_LLM_MODE must be one of: api, codex")
    normalized_mode = value.strip().lower()
    if normalized_mode not in {"api", "codex"}:
        raise ModelConfigurationError("UXA_LLM_MODE must be one of: api, codex")
    return cast(Literal["api", "codex"], normalized_mode)


def _reasoning_effort(
    settings: OpenAICompatibleSettings, role: ModelRole
) -> str | None:
    role_value = ModelRole(role)
    if role_value is ModelRole.COGNITIVE:
        return settings.cognitive_reasoning_effort
    if role_value in _REPORT_ROLES:
        return settings.report_reasoning_effort
    return settings.scent_reasoning_effort


def _absolute_attachment_path(attachment: ModelAttachment) -> Path:
    path = attachment.path
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def _safe_attachment_audit_path(path: Path) -> str:
    relative = Path(path.name) if path.is_absolute() else path
    if not relative.parts or any(
        part in {".", ".."} or "\x00" in part for part in relative.parts
    ):
        raise ValueError("attachment path must be safe and relative for audit")
    return PurePosixPath(*relative.parts).as_posix()


def _attachment_audit_metadata(attachment: ModelAttachment) -> dict[str, str]:
    return {
        "evidence_id": attachment.evidence_id,
        "path": _safe_attachment_audit_path(attachment.path),
        "media_type": attachment.media_type,
        "sha256": attachment.sha256,
    }


def _validated_attachment_bytes(attachment: ModelAttachment) -> bytes:
    path = _absolute_attachment_path(attachment)
    try:
        secure_assert_ancestors(path, "model attachment")
        content = secure_read_bytes(
            path,
            "model attachment",
            max_bytes=_MAX_MODEL_ATTACHMENT_BYTES,
        )
    except (OSError, RuntimeError) as error:
        raise ValueError("model attachment path or size is invalid") from error

    try:
        with Image.open(BytesIO(content)) as image:
            image_format = image.format
            image.verify()
    except Exception as error:
        raise ValueError("model attachment media is invalid") from error

    actual_media_type = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
    }.get(image_format or "")
    if actual_media_type != attachment.media_type:
        raise ValueError("model attachment media type does not match content")
    digest = hashlib.sha256(content).hexdigest()
    if digest != attachment.sha256:
        raise ValueError("model attachment checksum mismatch")
    return content


def _validate_model_attachments(messages: Sequence[ChatMessage]) -> None:
    for message in messages:
        for attachment in message.attachments:
            _validated_attachment_bytes(attachment)


def _audit_message_dump(message: ChatMessage) -> dict[str, object]:
    payload = message.model_dump()
    if message.attachments:
        payload["attachments"] = [
            _attachment_audit_metadata(attachment) for attachment in message.attachments
        ]
    return payload


def _http_message_payload(message: ChatMessage) -> dict[str, object]:
    if not message.attachments:
        return message.model_dump()
    parts: list[dict[str, object]] = [
        {"type": "text", "text": message.content},
    ]
    for attachment in message.attachments:
        content = _validated_attachment_bytes(attachment)
        encoded = base64.b64encode(content).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{attachment.media_type};base64,{encoded}"},
            }
        )
    return {"role": message.role, "content": parts}


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        delay = parsed.timestamp() - time.time()
    if delay < 0 or not math.isfinite(delay):
        return None
    return delay


@asynccontextmanager
async def _model_call_slot(
    limiter: asyncio.Semaphore | None,
) -> AsyncGenerator[None, None]:
    if limiter is None:
        yield
        return
    async with limiter:
        yield


@dataclass(frozen=True, slots=True, repr=False)
class OpenAICompatibleSettings:
    """Validated model endpoint settings loaded from environment variables."""

    base_url: str
    api_key: str = field(repr=False)
    scent_model: str
    cognitive_model: str
    mode: Literal["api", "codex"] = "api"
    scent_reasoning_effort: str | None = None
    cognitive_reasoning_effort: str | None = None
    timeout_seconds: float | None = 30.0
    max_concurrent_calls: int = 2
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    redaction_values: tuple[str, ...] = ()
    report_model: str | None = None
    report_reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        normalized_mode = _normalize_llm_mode(self.mode)
        object.__setattr__(self, "mode", normalized_mode)
        if normalized_mode == "api":
            normalized_url = self.base_url.rstrip("/")
            if not normalized_url:
                raise ModelConfigurationError("base_url must not be empty")
            _endpoint_origin(normalized_url)
            required_values = (("api_key", self.api_key),)
        else:
            normalized_url = ""
            required_values = ()
        for name, value in (
            *required_values,
            ("scent_model", self.scent_model),
            ("cognitive_model", self.cognitive_model),
        ):
            if not value:
                raise ModelConfigurationError(f"{name} must not be empty")
        if self.report_model is not None and not self.report_model:
            raise ModelConfigurationError("report_model must not be empty")
        for name in (
            "scent_reasoning_effort",
            "cognitive_reasoning_effort",
            "report_reasoning_effort",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_reasoning_effort(getattr(self, name), name=name),
            )
        if self.timeout_seconds is None and normalized_mode != "codex":
            raise ModelConfigurationError(
                "unbounded timeout is supported only in codex mode"
            )
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ModelConfigurationError("timeout_seconds must be greater than zero")
        if self.max_concurrent_calls < 1:
            raise ModelConfigurationError(
                "max_concurrent_calls must be greater than zero"
            )
        object.__setattr__(self, "base_url", normalized_url)
        object.__setattr__(self, "redaction_values", tuple(self.redaction_values))

    @property
    def endpoint_origin(self) -> str:
        if self.mode == "codex":
            return "codex-cli"
        return _endpoint_origin(self.base_url)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        dotenv_path: Path | None = None,
        report_synthesis_enabled: bool = False,
    ) -> OpenAICompatibleSettings:
        if environ is None:
            merged_values = dict(os.environ)
            load_environment_file(dotenv_path, environ=merged_values)
            values: Mapping[str, str] = merged_values
        else:
            merged_values = dict(environ)
            if dotenv_path is not None:
                load_environment_file(dotenv_path, environ=merged_values)
            values = merged_values
        mode = _normalize_llm_mode(values.get("UXA_LLM_MODE", "api"))
        names = (
            (
                (
                    "UXA_LLM_BASE_URL",
                    "UXA_LLM_API_KEY",
                )
                if mode == "api"
                else ()
            )
            + (
                "UXA_SCENT_MODEL",
                "UXA_COGNITIVE_MODEL",
            )
            + (("UXA_REPORT_MODEL",) if report_synthesis_enabled else ())
        )
        missing = [name for name in names if not values.get(name)]
        if missing:
            raise ModelConfigurationError(
                "missing model environment variables: " + ", ".join(missing)
            )
        if mode == "api":
            base_url = values["UXA_LLM_BASE_URL"]
            api_key = values["UXA_LLM_API_KEY"]
        else:
            base_url = ""
            api_key = ""
        timeout_seconds = _timeout_seconds(
            values.get("UXA_LLM_TIMEOUT_SECONDS", "30"), mode=mode
        )
        max_concurrent_calls = _as_int(
            values.get("UXA_LLM_MAX_CONCURRENT_CALLS", "2"),
            name="max_concurrent_calls",
        )
        return cls(
            base_url=base_url,
            api_key=api_key,
            scent_model=values["UXA_SCENT_MODEL"],
            cognitive_model=values["UXA_COGNITIVE_MODEL"],
            report_model=values.get("UXA_REPORT_MODEL") or None,
            mode=mode,
            scent_reasoning_effort=(
                values.get("UXA_LLM_SCENT_REASONING_EFFORT") or None
            ),
            cognitive_reasoning_effort=(
                values.get("UXA_LLM_COGNITIVE_REASONING_EFFORT") or None
            ),
            report_reasoning_effort=(
                values.get("UXA_LLM_REPORT_REASONING_EFFORT") or None
            ),
            timeout_seconds=timeout_seconds,
            max_concurrent_calls=max_concurrent_calls,
        )

    @classmethod
    def model_validate(
        cls,
        value: Mapping[str, object],
        *,
        report_synthesis_enabled: bool = False,
    ) -> OpenAICompatibleSettings:
        """Small Pydantic-like constructor useful at config boundaries/tests."""

        retry_value = value.get("retry_policy", RetryPolicy())
        if isinstance(retry_value, Mapping):
            retry_mapping = cast(Mapping[object, object], retry_value)
            retry = RetryPolicy(
                max_attempts=_as_int(
                    retry_mapping.get("max_attempts", 3), name="max_attempts"
                ),
                base_delay_seconds=_as_float(
                    retry_mapping.get("base_delay_seconds", 0.25),
                    name="base_delay_seconds",
                ),
                max_delay_seconds=_as_float(
                    retry_mapping.get("max_delay_seconds", 2.0),
                    name="max_delay_seconds",
                ),
                multiplier=_as_float(
                    retry_mapping.get("multiplier", 2.0), name="multiplier"
                ),
            )
        else:
            retry = cast(RetryPolicy, retry_value)
        redaction_value = value.get("redaction_values", ())
        if not isinstance(redaction_value, Sequence) or isinstance(
            redaction_value, (str, bytes)
        ):
            raise ModelConfigurationError("redaction_values must be a sequence")
        mode = _normalize_llm_mode(value.get("mode", "api"))
        timeout_seconds = _timeout_seconds(
            value.get("timeout_seconds", 30.0), mode=mode
        )
        max_concurrent_calls = _as_int(
            value.get("max_concurrent_calls", 2), name="max_concurrent_calls"
        )
        if mode == "api":
            base_url = str(value["base_url"])
            api_key = str(value["api_key"])
        else:
            base_url = ""
            api_key = ""
        report_model_value = value.get("report_model")
        if report_synthesis_enabled and not report_model_value:
            raise ModelConfigurationError(
                "report_model is required for report synthesis"
            )
        return cls(
            base_url=base_url,
            api_key=api_key,
            scent_model=str(value["scent_model"]),
            cognitive_model=str(value["cognitive_model"]),
            report_model=(
                None if report_model_value is None else str(report_model_value)
            ),
            mode=mode,
            scent_reasoning_effort=(
                None
                if value.get("scent_reasoning_effort") is None
                else str(value["scent_reasoning_effort"])
            ),
            cognitive_reasoning_effort=(
                None
                if value.get("cognitive_reasoning_effort") is None
                else str(value["cognitive_reasoning_effort"])
            ),
            report_reasoning_effort=(
                None
                if value.get("report_reasoning_effort") is None
                else str(value["report_reasoning_effort"])
            ),
            timeout_seconds=timeout_seconds,
            max_concurrent_calls=max_concurrent_calls,
            retry_policy=retry,
            redaction_values=tuple(
                str(item) for item in cast(Sequence[object], redaction_value)
            ),
        )

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleSettings("
            f"base_url={self.base_url!r}, "
            f"scent_model={self.scent_model!r}, "
            f"cognitive_model={self.cognitive_model!r}, "
            f"report_model={self.report_model!r}, "
            f"scent_reasoning_effort={self.scent_reasoning_effort!r}, "
            f"cognitive_reasoning_effort={self.cognitive_reasoning_effort!r}, "
            f"report_reasoning_effort={self.report_reasoning_effort!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_concurrent_calls={self.max_concurrent_calls!r}, "
            f"retry_policy={self.retry_policy!r})"
        )

    def model_for_role(self, role: ModelRole) -> str:
        role_value = ModelRole(role)
        if role_value in _REPORT_ROLES:
            if self.report_model is None:
                raise ModelConfigurationError(
                    "report_model is required for report roles"
                )
            return self.report_model
        if role_value in {ModelRole.COARSE_SCENT, ModelRole.FULL_SCENT}:
            return self.scent_model
        return self.cognitive_model


def _schema_version(schema: type[BaseModel]) -> str:
    value = getattr(schema, "schema_version", schema.__name__)
    return value if isinstance(value, str) and value else schema.__name__


def _normalize_messages(
    messages: Sequence[ChatMessage | Mapping[str, object]],
) -> tuple[ChatMessage, ...]:
    normalized: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            normalized.append(message)
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("model messages need string role and content")
        attachments_value = message.get("attachments", ())
        if not isinstance(attachments_value, Sequence) or isinstance(
            attachments_value, (str, bytes)
        ):
            raise ValueError("model message attachments must be a sequence")
        normalized.append(
            ChatMessage(
                role=role,
                content=content,
                attachments=tuple(cast(Sequence[ModelAttachment], attachments_value)),
            )
        )
    if not normalized:
        raise ValueError("structured model call needs at least one message")
    return tuple(normalized)


def _prompt_digest(messages: Sequence[ChatMessage]) -> str:
    canonical = json.dumps(
        [_audit_message_dump(message) for message in messages],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _usage(payload: Mapping[str, object]) -> TokenUsage:
    value = payload.get("usage")
    if not isinstance(value, Mapping):
        return TokenUsage()
    usage = cast(Mapping[object, object], value)
    return TokenUsage(
        prompt_tokens=_as_int(usage.get("prompt_tokens", 0) or 0, name="prompt_tokens"),
        completion_tokens=_as_int(
            usage.get("completion_tokens", 0) or 0, name="completion_tokens"
        ),
        total_tokens=_as_int(usage.get("total_tokens", 0) or 0, name="total_tokens"),
    )


async def _bounded_http_response(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Mapping[str, str],
    content: bytes,
) -> httpx.Response:
    request = client.build_request("POST", url, headers=headers, content=content)
    response = await client.send(request, stream=True)
    chunks: list[bytes] = []
    total = 0
    try:
        if response.is_stream_consumed:
            total = len(response.content)
            if total > _MAX_MODEL_RESPONSE_BYTES:
                raise _ModelResponseTooLarge("response-too-large")
            chunks.append(response.content)
        else:
            async for chunk in response.aiter_raw():
                total += len(chunk)
                if total > _MAX_MODEL_RESPONSE_BYTES:
                    raise _ModelResponseTooLarge("response-too-large")
                chunks.append(chunk)
    finally:
        await response.aclose()
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        content=b"".join(chunks),
        request=request,
        extensions=response.extensions,
    )


def _response_body(response: httpx.Response) -> object:
    if len(response.content) > _MAX_MODEL_RESPONSE_BYTES:
        raise _ModelResponseTooLarge("response-too-large")
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


def _provider_error_metadata(
    response: httpx.Response, body: object
) -> dict[str, object]:
    metadata: dict[str, object] = {"status_code": response.status_code}
    for header_name in ("x-request-id", "request-id"):
        request_id = _safe_provider_text(response.headers.get(header_name))
        if request_id is not None:
            metadata["request_id"] = request_id
            break
    if not isinstance(body, Mapping):
        return metadata
    body_mapping = cast(Mapping[object, object], body)
    error_value = body_mapping.get("error")
    if not isinstance(error_value, Mapping):
        return metadata
    error = cast(Mapping[object, object], error_value)
    for source_name, target_name in (("code", "error_code"), ("type", "error_type")):
        value = _safe_provider_text(error.get(source_name))
        if value is not None:
            metadata[target_name] = value
    return metadata


def _provider_error_code(body: object) -> str | None:
    if not isinstance(body, Mapping):
        return None
    body_mapping = cast(Mapping[object, object], body)
    error_value = body_mapping.get("error")
    if not isinstance(error_value, Mapping):
        return None
    return _safe_provider_text(cast(Mapping[object, object], error_value).get("code"))


def _provider_failure_reason(status_code: int, body: object) -> str:
    error_code = _provider_error_code(body)
    normalized_code = error_code.upper() if error_code is not None else ""
    if normalized_code in {
        "MODEL_UNAVAILABLE",
        "MODEL_NOT_AVAILABLE",
        "MODEL_NOT_FOUND",
    }:
        return "model unavailable"
    if status_code == 429:
        return "rate limit"
    if 500 <= status_code <= 599:
        return "server error"
    return "request rejected"


def _body_text(body: object) -> str:
    if isinstance(body, str):
        return body
    return json.dumps(body, ensure_ascii=True, sort_keys=True)


def _is_schema_unsupported(status_code: int, body: object) -> bool:
    if status_code != 400:
        return False
    text = _body_text(body).lower()
    if "invalid_request" in text or "invalid request" in text:
        return True
    return any(
        marker in text
        for marker in ("response_format", "json_schema", "strict", "unsupported")
    )


def _is_safety_rejection(status_code: int, body: object) -> bool:
    text = _body_text(body).lower()
    return status_code in {400, 403} and any(
        marker in text
        for marker in (
            "safety",
            "content_policy",
            "policy violation",
            "blocked content",
        )
    )


def _transport_error_category(error: httpx.TransportError) -> str:
    """Return a stable category without persisting endpoint exception text."""

    categories: tuple[tuple[type[BaseException], str], ...] = (
        (httpx.ConnectTimeout, "connect-timeout"),
        (httpx.ReadTimeout, "read-timeout"),
        (httpx.WriteTimeout, "write-timeout"),
        (httpx.PoolTimeout, "pool-timeout"),
        (httpx.ConnectError, "connect-error"),
        (httpx.RemoteProtocolError, "protocol-error"),
        (httpx.ReadError, "read-error"),
        (httpx.WriteError, "write-error"),
        (httpx.CloseError, "close-error"),
    )
    for error_type, category in categories:
        if isinstance(error, error_type):
            return category
    if isinstance(error, httpx.TimeoutException):
        return "transport-timeout"
    return "transport-error"


_MISSING_RESPONSE_CONTENT = object()


def _response_message_content(body: object) -> object:
    if not isinstance(body, Mapping):
        raise ValueError("response body is not an object")
    response_mapping = cast(Mapping[object, object], body)
    choices_value = response_mapping.get("choices")
    choices = cast(object, choices_value)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise ValueError("response has no choices")
    choice_items = cast(Sequence[object], choices)
    if not choice_items or not isinstance(choice_items[0], Mapping):
        raise ValueError("response has no first choice")
    choice = cast(Mapping[object, object], choice_items[0])
    message_value = choice.get("message")
    message = cast(object, message_value)
    if not isinstance(message, Mapping):
        raise ValueError("response choice has no message")
    message_mapping = cast(Mapping[object, object], message)
    return message_mapping.get("content", _MISSING_RESPONSE_CONTENT)


def _text_content_stream(content: object) -> str | None:
    if isinstance(content, str):
        if len(content) > _MAX_MODEL_RESPONSE_BYTES:
            raise _ModelResponseTooLarge("response-too-large")
        return content
    if isinstance(content, Mapping):
        content_mapping = cast(Mapping[object, object], content)
        if "type" not in content_mapping:
            return None
        part_type = content_mapping.get("type")
        text = content_mapping.get("text")
        if part_type not in _SAFE_TEXT_PART_TYPES or not isinstance(text, str):
            raise ValueError("response message has no unambiguous text content")
        if len(text) > _MAX_MODEL_RESPONSE_BYTES:
            raise _ModelResponseTooLarge("response-too-large")
        return text
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise ValueError("response message has no JSON content")
    parts: list[str] = []
    total_length = 0
    for part in cast(Sequence[object], content):
        if not isinstance(part, Mapping):
            raise ValueError("response message has ambiguous content parts")
        part_mapping = cast(Mapping[object, object], part)
        part_type = part_mapping.get("type")
        text = part_mapping.get("text")
        if part_type not in _SAFE_TEXT_PART_TYPES or not isinstance(text, str):
            raise ValueError("response message has ambiguous content parts")
        parts.append(text)
        total_length += len(text)
        if total_length > _MAX_MODEL_RESPONSE_BYTES:
            raise _ModelResponseTooLarge("response-too-large")
    if not parts:
        raise ValueError("response message has no JSON content")
    return "".join(parts)


def _json_decoder() -> json.JSONDecoder:
    def reject_json_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    return json.JSONDecoder(parse_constant=reject_json_constant)


def _strict_json_object(content_text: str) -> dict[str, object]:
    stripped = content_text.strip()
    if not stripped:
        raise ValueError("response message has no JSON content")
    parsed, end = _json_decoder().raw_decode(stripped)
    if end != len(stripped) or not isinstance(parsed, Mapping):
        raise ValueError("structured response must be one JSON object")
    return dict(cast(Mapping[str, object], parsed))


def _json_container_spans(content_text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escaped = False
    for index, character in enumerate(content_text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"' and stack:
            in_string = True
            continue
        if character in "[{":
            if not stack:
                start = index
            stack.append(character)
            continue
        if character not in "]}" or not stack:
            continue
        expected = "[" if character == "]" else "{"
        if stack[-1] != expected:
            stack.clear()
            start = None
            continue
        stack.pop()
        if not stack and start is not None:
            spans.append((start, index + 1))
            start = None
    return tuple(spans)


def _report_json_object(content_text: str) -> dict[str, object]:
    try:
        return _strict_json_object(content_text)
    except (TypeError, ValueError):
        pass

    decoder = _json_decoder()
    valid_values: list[tuple[int, int, object]] = []
    for start, end in _json_container_spans(content_text):
        try:
            parsed, parsed_end = decoder.raw_decode(content_text[start:end])
        except (TypeError, ValueError):
            continue
        if parsed_end == end - start:
            valid_values.append((start, end, parsed))
    object_values = [
        value for value in valid_values if isinstance(value[2], Mapping)
    ]
    if len(object_values) == 1 and len(valid_values) > 1:
        raise ValueError("structured response contains multiple JSON values")
    if len(object_values) != 1 or len(valid_values) != 1:
        raise ValueError("structured response must contain exactly one JSON object")

    start, end, parsed = object_values[0]
    envelope = content_text[:start] + (" " * (end - start)) + content_text[end:]
    if "[" in envelope or "]" in envelope or _REPORT_JSON_SCALAR.search(envelope):
        raise ValueError("structured response contains multiple JSON values")
    return dict(cast(Mapping[str, object], parsed))


def _structured_content(
    body: object,
    *,
    role: ModelRole | None = None,
) -> object:
    content = _response_message_content(body)
    if isinstance(content, Mapping):
        content_mapping = cast(Mapping[object, object], content)
        if _text_content_stream(content_mapping) is None:
            return cast(dict[str, object], content_mapping)
    content_text = _text_content_stream(cast(object, content))
    if content_text is None:
        raise ValueError("response message has no JSON content")
    if role in _REPORT_ROLES:
        return _report_json_object(content_text)
    return _strict_json_object(content_text)


def _safe_finish_reason(body: object) -> str | None:
    if not isinstance(body, Mapping):
        return None
    body_mapping = cast(Mapping[object, object], body)
    choices_value = body_mapping.get("choices")
    if not isinstance(choices_value, Sequence) or isinstance(
        choices_value, (str, bytes)
    ):
        return None
    choices = cast(Sequence[object], choices_value)
    if not choices or not isinstance(choices[0], Mapping):
        return None
    choice = cast(Mapping[object, object], choices[0])
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason in _SAFE_FINISH_REASONS:
        return finish_reason
    return None


def _structural_value_type(value: object) -> str:
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return "other"


def _response_content_length(content: object) -> int:
    if isinstance(content, str):
        return min(len(content), _MAX_DIAGNOSTIC_CONTENT_LENGTH)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        total = 0
        for item in cast(Sequence[object], content):
            if isinstance(item, Mapping):
                text = cast(Mapping[object, object], item).get("text")
                if isinstance(text, str):
                    total += len(text)
        return min(total, _MAX_DIAGNOSTIC_CONTENT_LENGTH)
    if isinstance(content, Mapping):
        text = cast(Mapping[object, object], content).get("text")
        if isinstance(text, str):
            return min(len(text), _MAX_DIAGNOSTIC_CONTENT_LENGTH)
    return 0


def _response_content_markers(content: object) -> list[str]:
    if content is _MISSING_RESPONSE_CONTENT:
        return ["missing"]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = cast(Sequence[object], content)
        text_parts = 0
        non_text_parts = 0
        for item in parts:
            if not isinstance(item, Mapping):
                non_text_parts += 1
                continue
            part = cast(Mapping[object, object], item)
            part_type = part.get("type")
            if isinstance(part.get("text"), str) and (
                part_type is None
                or (isinstance(part_type, str) and part_type in _SAFE_TEXT_PART_TYPES)
            ):
                text_parts += 1
            else:
                non_text_parts += 1
        markers = ["content_parts"]
        if text_parts:
            markers.append("text_only_parts" if not non_text_parts else "mixed_parts")
        if len(parts) > 1:
            markers.append("multiple_parts")
        if not parts:
            markers.append("empty")
        return markers
    if isinstance(content, Mapping):
        content_mapping = cast(Mapping[object, object], content)
        if isinstance(content_mapping.get("text"), str):
            return ["text_part"]
        return ["structured_object"]
    if not isinstance(content, str):
        return [_structural_value_type(content)]
    stripped = content.strip()
    if not stripped:
        return ["empty"]
    markers: list[str] = []
    if "```" in content:
        markers.append("markdown_fence")
    first_object = content.find("{")
    if first_object >= 0:
        markers.append("object_candidate")
        if content[:first_object].strip():
            markers.append("prose_prefix")
        if content[first_object:].lstrip().startswith("{") and stripped.startswith("{"):
            markers.append("raw_object_prefix")
    if stripped.startswith("["):
        markers.append("array_prefix")
    if stripped[0] in '-0123456789tfn"':
        markers.append("scalar_prefix")
    last_object = content.rfind("}")
    if last_object >= 0 and content[last_object + 1 :].strip():
        markers.append("prose_suffix")
    return markers or ["prose"]


def _response_content_diagnostics(body: object) -> dict[str, object]:
    try:
        content = _response_message_content(body)
    except (TypeError, ValueError):
        content = _MISSING_RESPONSE_CONTENT
    diagnostics: dict[str, object] = {
        "response_content_type": (
            "missing"
            if content is _MISSING_RESPONSE_CONTENT
            else _structural_value_type(content)
        ),
        "response_content_length": _response_content_length(content),
        "response_content_markers": _response_content_markers(content),
    }
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        diagnostics["response_content_part_count"] = len(
            cast(Sequence[object], content)
        )
    return diagnostics


def _schema_field_names(schema: type[BaseModel]) -> frozenset[str]:
    names: set[str] = set()
    pending: list[object] = [schema.model_json_schema()]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            properties = mapping.get("properties")
            if isinstance(properties, Mapping):
                names.update(
                    name
                    for name in cast(Mapping[object, object], properties)
                    if isinstance(name, str) and _SAFE_STRUCTURAL_NAME.fullmatch(name)
                )
            pending.extend(mapping.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            pending.extend(cast(Sequence[object], value))
    return frozenset(names)


def _safe_structural_name(
    value: object,
    *,
    allowed_names: frozenset[str],
) -> str:
    if isinstance(value, str) and value in allowed_names:
        return value
    if isinstance(value, str):
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        return f"unknown-{digest[:12]}"
    return "[redacted]"


def _safe_validation_path(
    value: object,
    *,
    allowed_names: frozenset[str],
) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ["[redacted]"]
    path: list[object] = []
    for item in cast(Sequence[object], value):
        if type(item) is int and item >= 0:
            path.append(item)
        else:
            path.append(_safe_structural_name(item, allowed_names=allowed_names))
    return path


def _structured_output_diagnostics(
    *,
    role: ModelRole,
    mode: str,
    attempts: int,
    schema: type[BaseModel],
    parsed: object,
    stage: str,
    finish_reason: str | None = None,
    error: BaseException | None = None,
    body: object = _MISSING_RESPONSE_CONTENT,
) -> dict[str, object]:
    allowed_names = _schema_field_names(schema)
    diagnostics: dict[str, object] = {
        "role": role.value,
        "response_mode": mode,
        "attempt_count": attempts,
        "stage": stage,
        "top_level_keys": [],
        "top_level_value_types": {},
    }
    if body is not _MISSING_RESPONSE_CONTENT:
        diagnostics.update(_response_content_diagnostics(body))
    if finish_reason is not None:
        diagnostics["finish_reason"] = finish_reason
    if isinstance(parsed, Mapping):
        parsed_mapping = cast(Mapping[object, object], parsed)
        safe_items = sorted(
            (
                _safe_structural_name(key, allowed_names=allowed_names),
                _structural_value_type(value),
            )
            for key, value in parsed_mapping.items()
        )
        diagnostics["top_level_keys"] = [key for key, _ in safe_items[:64]]
        diagnostics["top_level_value_types"] = {
            key: value for key, value in safe_items[:64]
        }
    else:
        diagnostics["parsed_type"] = _structural_value_type(parsed)
    if isinstance(error, ValidationError):
        errors: list[dict[str, object]] = []
        for item in error.errors()[:16]:
            errors.append(
                {
                    "path": _safe_validation_path(
                        item.get("loc"), allowed_names=allowed_names
                    ),
                    "type": (
                        item_type
                        if _SAFE_STRUCTURAL_NAME.fullmatch(
                            item_type := item.get("type")
                        )
                        else "[redacted]"
                    ),
                }
            )
        diagnostics["validation_errors"] = errors
    elif error is not None:
        diagnostics["error_type"] = type(error).__name__
    return diagnostics


def _normalize_report_output(role: ModelRole, parsed: object) -> object:
    """Normalize the provider's generic findings key before local validation."""

    if role not in _REPORT_ROLES or not isinstance(parsed, Mapping):
        return parsed
    parsed_mapping = cast(Mapping[object, object], parsed)
    target_field = {
        ModelRole.REPORT_ANALYST: "candidate_findings",
        ModelRole.REPORT_ADJUDICATOR: "final_findings",
    }.get(role)
    if target_field is None or "findings" not in parsed_mapping:
        return parsed_mapping
    if target_field in parsed_mapping:
        raise ValueError("report response contains conflicting findings fields")
    normalized = dict(parsed_mapping)
    normalized[target_field] = normalized.pop("findings")
    return normalized


def _response_mode(role: ModelRole) -> str:
    role_value = ModelRole(role)
    # This provider rejects response_format on the report route. Keep JSON
    # parsing and Pydantic validation local after receiving plain content.
    if role_value in _REPORT_ROLES:
        return "plain"
    if role_value is ModelRole.COGNITIVE:
        return "json-object"
    return "strict"


class _StructuredCallSupport:
    """Shared safe records, manifests, retry events, and delay mechanics."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        *,
        provider_id: str,
        provider_version: str,
    ) -> None:
        self.settings = settings
        self._provider_id = provider_id
        self._provider_version = provider_version
        self._records: list[ModelCallRecord] = []
        self._retry_events: list[RetryEvent] = []

    @property
    def endpoint_origin(self) -> str:
        return self.settings.endpoint_origin

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def records(self) -> tuple[ModelCallRecord, ...]:
        return tuple(self._records)

    @property
    def retry_events(self) -> tuple[RetryEvent, ...]:
        return tuple(self._retry_events)

    def manifest(
        self,
        role: ModelRole,
        model: str,
        *,
        prompt_version: str | None = None,
        schema_version: str = "unknown",
    ) -> ModelManifest:
        role_value = ModelRole(role)
        default_prompt = {
            ModelRole.COARSE_SCENT: "scent-coarse-v1",
            ModelRole.FULL_SCENT: "scent-full-v1",
            ModelRole.COGNITIVE: "cognitive-v1",
            ModelRole.REPORT_ANALYST: "report-analyst-v3",
            ModelRole.REPORT_EVIDENCE_AUDITOR: "report-evidence-auditor-v3",
            ModelRole.REPORT_PATTERN_REVIEWER: "report-pattern-reviewer-v3",
            ModelRole.REPORT_ADJUDICATOR: "report-adjudicator-v3",
        }[role_value]
        return ModelManifest(
            provider_id=self._provider_id,
            role=role_value,
            model_id=model,
            endpoint_origin=self.endpoint_origin,
            prompt_version=prompt_version or default_prompt,
            schema_version=schema_version,
            provider_version=self._provider_version,
        )

    def _retry(
        self,
        role: ModelRole,
        model: str,
        attempt: int,
        reason: str,
        status_code: int | None,
        policy: RetryPolicy,
        retry_number: int,
        delay_override: float | None = None,
    ) -> RetryEvent:
        event = RetryEvent(
            role=role,
            model=model,
            attempt=attempt,
            reason=reason,
            status_code=status_code,
            delay_seconds=max(
                policy.delay_for_retry(retry_number), delay_override or 0.0
            ),
        )
        self._retry_events.append(event)
        return event

    async def _sleep(self, delay_seconds: float) -> None:
        if delay_seconds:
            await asyncio.sleep(delay_seconds)

    def _record(
        self,
        role: ModelRole,
        model: str,
        prompt_digest: str,
        schema_version: str,
        attempts: int,
        started: float,
        request_payload: Mapping[str, object],
        response_payload: Mapping[str, object],
        token_usage: TokenUsage,
        retries: Sequence[RetryEvent],
    ) -> ModelCallRecord:
        record = ModelCallRecord(
            role=role,
            model=model,
            endpoint_origin=self.endpoint_origin,
            prompt_digest=prompt_digest,
            schema_version=schema_version,
            attempts=max(attempts, 1),
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            token_usage=token_usage,
            request=cast(
                dict[str, Any],
                sanitize_for_log(
                    request_payload,
                    secrets=(self.settings.api_key, *self.settings.redaction_values),
                ),
            ),
            response=cast(
                dict[str, Any],
                sanitize_for_log(
                    response_payload,
                    secrets=(self.settings.api_key, *self.settings.redaction_values),
                ),
            ),
            retries=tuple(retries),
        )
        self._records.append(record)
        logger.info(
            "structured model call role=%s model=%s endpoint_origin=%s attempts=%d",
            record.role.value,
            record.model,
            record.endpoint_origin,
            record.attempts,
        )
        return record


class OpenAICompatibleStructuredClient(_StructuredCallSupport):
    """HTTP adapter with local schema validation and bounded safe retries."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
        call_limiter: asyncio.Semaphore | None = None,
    ) -> None:
        super().__init__(
            settings,
            provider_id="openai-compatible-structured",
            provider_version="openai-compatible-v1",
        )
        self._http_client = http_client or httpx.AsyncClient(
            timeout=settings.timeout_seconds
        )
        self._call_limiter = call_limiter

    async def complete(
        self,
        schema: type[SchemaT],
        messages: Sequence[ChatMessage | Mapping[str, object]],
        model: str,
        role: ModelRole,
    ) -> SchemaT:
        normalized_messages = _normalize_messages(messages)
        _validate_model_attachments(normalized_messages)
        role_value = ModelRole(role)
        prompt_digest = _prompt_digest(normalized_messages)
        schema_version = _schema_version(schema)
        retry_policy = self.settings.retry_policy
        retries: list[RetryEvent] = []
        attempts = 0
        started = time.perf_counter()
        response_payload: object = {}
        # Keep local validation while using the provider-compatible object mode
        # for report and cognitive roles.
        mode = _response_mode(role_value)
        last_reason = "model call failed"
        last_provider_metadata: dict[str, object] = {}
        last_structural_diagnostics: dict[str, object] = {}

        while attempts < retry_policy.max_attempts:
            attempts += 1
            request_payload = self._request_payload(
                schema, normalized_messages, model, role_value, mode
            )
            request_body = serialize_transport_json(request_payload)
            if role_value in _REPORT_ROLES:
                enforce_transport_size(request_body)
            try:
                async with _model_call_slot(self._call_limiter):
                    response = await _bounded_http_response(
                        self._http_client,
                        f"{self.settings.base_url}/chat/completions",
                        headers={
                            "accept": "application/json",
                            "content-type": "application/json",
                            "authorization": f"Bearer {self.settings.api_key}",
                        },
                        content=request_body,
                    )
                response_payload = _response_body(response)
                last_provider_metadata = (
                    _provider_error_metadata(response, response_payload)
                    if not 200 <= response.status_code < 300
                    else {}
                )
            except _ModelResponseTooLarge:
                last_reason = "response-too-large"
                last_provider_metadata = {}
                break
            except httpx.TransportError as error:
                last_reason = _transport_error_category(error)
                last_provider_metadata = {}
                if attempts >= retry_policy.max_attempts:
                    break
                retries.append(
                    self._retry(
                        role_value,
                        model,
                        attempts,
                        last_reason,
                        None,
                        retry_policy,
                        len(retries) + 1,
                    )
                )
                await self._sleep(retries[-1].delay_seconds)
                continue

            if _is_safety_rejection(response.status_code, response_payload):
                last_reason = "safety rejection"
                break
            if response.status_code in {401, 403}:
                last_reason = "authentication failure"
                break
            if (
                _is_schema_unsupported(response.status_code, response_payload)
                and mode == "strict"
            ):
                mode = "json-object"
                continue
            if response.status_code == 429:
                last_reason = "rate limit"
                if attempts < retry_policy.max_attempts:
                    retries.append(
                        self._retry(
                            role_value,
                            model,
                            attempts,
                            "rate-limit",
                            response.status_code,
                            retry_policy,
                            len(retries) + 1,
                            delay_override=_retry_after_seconds(response),
                        )
                    )
                    await self._sleep(retries[-1].delay_seconds)
                    continue
                break
            if 500 <= response.status_code <= 599:
                last_reason = _provider_failure_reason(
                    response.status_code, response_payload
                )
                if attempts < retry_policy.max_attempts:
                    retries.append(
                        self._retry(
                            role_value,
                            model,
                            attempts,
                            "server-error",
                            response.status_code,
                            retry_policy,
                            len(retries) + 1,
                        )
                    )
                    await self._sleep(retries[-1].delay_seconds)
                    continue
                break
            if not 200 <= response.status_code < 300:
                last_reason = _provider_failure_reason(
                    response.status_code, response_payload
                )
                break

            parsed: object = {}
            diagnostic_stage = "content_parsing"
            finish_reason = _safe_finish_reason(response_payload)
            try:
                if finish_reason == "length":
                    raise ValueError("response content truncated by provider")
                parsed = _structured_content(response_payload, role=role_value)
                diagnostic_stage = "normalization"
                parsed = _normalize_report_output(role_value, parsed)
                diagnostic_stage = "schema_validation"
                result = schema.model_validate(parsed)
            except _ModelResponseTooLarge:
                last_structural_diagnostics = _structured_output_diagnostics(
                    role=role_value,
                    mode=mode,
                    attempts=attempts,
                    schema=schema,
                    parsed=parsed,
                    stage=diagnostic_stage,
                    finish_reason=finish_reason,
                    body=response_payload,
                )
                last_reason = "response-too-large"
                break
            except ValidationError as error:
                last_structural_diagnostics = _structured_output_diagnostics(
                    role=role_value,
                    mode=mode,
                    attempts=attempts,
                    schema=schema,
                    parsed=parsed,
                    stage=diagnostic_stage,
                    finish_reason=finish_reason,
                    error=error,
                    body=response_payload,
                )
                last_reason = "invalid structured output"
                if attempts < retry_policy.max_attempts:
                    retries.append(
                        self._retry(
                            role_value,
                            model,
                            attempts,
                            "invalid-structured-output",
                            response.status_code,
                            retry_policy,
                            len(retries) + 1,
                        )
                    )
                    await self._sleep(retries[-1].delay_seconds)
                    continue
                break
            except (ValueError, TypeError) as error:
                last_structural_diagnostics = _structured_output_diagnostics(
                    role=role_value,
                    mode=mode,
                    attempts=attempts,
                    schema=schema,
                    parsed=parsed,
                    stage=diagnostic_stage,
                    finish_reason=finish_reason,
                    error=error,
                    body=response_payload,
                )
                last_reason = "invalid structured output"
                if attempts < retry_policy.max_attempts:
                    retries.append(
                        self._retry(
                            role_value,
                            model,
                            attempts,
                            "invalid-structured-output",
                            response.status_code,
                            retry_policy,
                            len(retries) + 1,
                        )
                    )
                    await self._sleep(retries[-1].delay_seconds)
                    continue
                break

            record = self._record(
                role_value,
                model,
                prompt_digest,
                schema_version,
                attempts,
                started,
                self._request_payload(
                    schema,
                    normalized_messages,
                    model,
                    role_value,
                    mode,
                    include_attachment_bytes=False,
                ),
                cast(Mapping[str, object], response_payload),
                (
                    _usage(cast(Mapping[str, object], response_payload))
                    if isinstance(response_payload, Mapping)
                    else TokenUsage()
                ),
                retries,
            )
            del record
            return result

        request_payload = self._request_payload(
            schema,
            normalized_messages,
            model,
            role_value,
            mode,
            include_attachment_bytes=False,
        )
        response_metadata: dict[str, object] = {"failure": last_reason}
        if last_provider_metadata:
            response_metadata["provider"] = last_provider_metadata
        if last_structural_diagnostics:
            response_metadata["diagnostics"] = last_structural_diagnostics
        self._record(
            role_value,
            model,
            prompt_digest,
            schema_version,
            attempts,
            started,
            request_payload,
            response_metadata,
            (
                _usage(cast(Mapping[str, object], response_payload))
                if isinstance(response_payload, Mapping)
                else TokenUsage()
            ),
            retries,
        )
        status_code = last_provider_metadata.get("status_code")
        error_code = last_provider_metadata.get("error_code")
        error_type = last_provider_metadata.get("error_type")
        request_id = last_provider_metadata.get("request_id")
        raise ModelFailureError(
            last_reason,
            status_code=status_code if type(status_code) is int else None,
            error_code=error_code if isinstance(error_code, str) else None,
            error_type=error_type if isinstance(error_type, str) else None,
            request_id=request_id if isinstance(request_id, str) else None,
            diagnostics=last_structural_diagnostics,
        )

    def _request_payload(
        self,
        schema: type[BaseModel],
        messages: Sequence[ChatMessage],
        model: str,
        role: ModelRole,
        mode: str,
        *,
        include_attachment_bytes: bool = True,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                (
                    _http_message_payload(message)
                    if include_attachment_bytes
                    else _audit_message_dump(message)
                )
                for message in messages
            ],
        }
        reasoning_effort = _reasoning_effort(self.settings, role)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if mode == "strict":
            schema_payload = schema.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_version(schema).replace("-", "_")
                    or "structured_response",
                    "strict": True,
                    "schema": schema_payload,
                },
            }
        elif mode == "json-object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def request_size(
        self,
        schema: type[BaseModel],
        messages: Sequence[ChatMessage],
        *,
        model: str,
        role: ModelRole,
    ) -> int:
        role_value = ModelRole(role)
        mode = _response_mode(role_value)
        return len(
            serialize_transport_json(
                self._request_payload(schema, messages, model, role_value, mode)
            )
        )


class _CodexAttemptError(RuntimeError):
    """Sanitized failure category from one Codex subprocess attempt."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


_CODEX_CLEANUP_TIMEOUT_SECONDS = 0.25
_CODEX_TERMINATION_GRACE_SECONDS = 0.05
_CODEX_PERMISSION_PROFILE = "uxa-evidence-only"
_CODEX_PERMISSION_CONFIG = (
    f"permissions.{_CODEX_PERMISSION_PROFILE}="
    '{filesystem={":root"="deny",":minimal"="read",'
    '":workspace_roots"={"."="write"}},network={enabled=false}}'
)


@dataclass(frozen=True, slots=True)
class _CodexCleanupResult:
    tree_terminated: bool
    communication_reaped: bool
    process_reaped: bool

    @property
    def success(self) -> bool:
        return (
            self.tree_terminated and self.communication_reaped and self.process_reaped
        )


def _codex_process_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_codex_process(process: asyncio.subprocess.Process) -> None:
    try:
        process.kill()
    except (OSError, RuntimeError):
        pass


def _codex_process_group_id(process: asyncio.subprocess.Process) -> int | None:
    if os.name == "nt":
        return None
    try:
        return os.getpgid(process.pid)
    except OSError:
        # start_new_session makes leader PID equal process-group ID.
        return process.pid


def _windows_descendant_pids(root_pid: int) -> tuple[int, ...] | None:
    if os.name != "nt":
        return ()
    import ctypes
    from ctypes import wintypes

    class _ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessEntry),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessEntry),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot in (None, invalid_handle):
        return None

    children: dict[int, list[int]] = {}
    entry = _ProcessEntry()
    entry.dwSize = ctypes.sizeof(_ProcessEntry)
    try:
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return None
        while True:
            children.setdefault(entry.th32ParentProcessID, []).append(
                entry.th32ProcessID
            )
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)

    descendants: list[int] = []
    pending = list(children.get(root_pid, ()))
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        descendants.append(pid)
        pending.extend(children.get(pid, ()))
    return tuple(descendants)


def _windows_terminate_process(pid: int, deadline: float) -> bool:
    if os.name != "nt":
        return True
    import ctypes
    from ctypes import wintypes

    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(0x00100001, False, pid)
    if not handle:
        return True
    try:
        kernel32.TerminateProcess(handle, 1)
        remaining_ms = max(
            0,
            int((deadline - asyncio.get_running_loop().time()) * 1000),
        )
        if remaining_ms == 0:
            return False
        reaped = kernel32.WaitForSingleObject(handle, remaining_ms) == 0
        return reaped
    finally:
        kernel32.CloseHandle(handle)


def _task_result(task: asyncio.Future[Any]) -> object:
    try:
        return task.result()
    except BaseException as error:
        return error


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    _task_result(task)


async def _wait_task_until(
    task: asyncio.Future[Any],
    deadline: float,
) -> tuple[bool, object]:
    if task.done():
        return True, _task_result(task)
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        return False, None
    done, _ = await asyncio.wait({task}, timeout=remaining)
    if task not in done:
        return False, None
    return True, _task_result(task)


async def _cancel_task_until(
    task: asyncio.Future[Any],
    deadline: float,
) -> tuple[bool, object]:
    if not task.done():
        task.cancel()
    if task.done():
        return True, _task_result(task)
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining > 0:
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if task in done:
            return True, _task_result(task)
    task.add_done_callback(_consume_task_result)
    return False, None


async def _shielded_wait_task_until(
    task: asyncio.Future[Any],
    deadline: float,
) -> tuple[bool, object]:
    wait_task = asyncio.create_task(_wait_task_until(task, deadline))
    while True:
        try:
            return await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            if asyncio.get_running_loop().time() >= deadline:
                await _cancel_task_until(wait_task, deadline)
                return False, None


async def _reap_process_until(
    process: asyncio.subprocess.Process,
    deadline: float,
    *,
    wait_task: asyncio.Future[Any] | None = None,
) -> bool:
    if wait_task is None:
        try:
            wait_task = asyncio.create_task(process.wait())
        except Exception:
            _kill_codex_process(process)
            return False
    completed, result = await _wait_task_until(wait_task, deadline)
    if completed and not isinstance(result, BaseException):
        return True
    if completed:
        _kill_codex_process(process)
        try:
            wait_task = asyncio.create_task(process.wait())
        except Exception:
            return False
    else:
        _kill_codex_process(process)
    completed, result = await _wait_task_until(wait_task, deadline)
    if completed and not isinstance(result, BaseException):
        return True
    if not completed:
        completed, result = await _cancel_task_until(wait_task, deadline)
    return completed and not isinstance(result, BaseException)


async def _terminate_codex_process_tree(
    process: asyncio.subprocess.Process,
    *,
    process_group_id: int | None,
    force: bool = False,
    deadline: float,
) -> bool:
    if os.name == "nt":
        if deadline <= asyncio.get_running_loop().time():
            _kill_codex_process(process)
            return False
        descendants_ok = True
        try:
            descendant_pids = _windows_descendant_pids(process.pid)
        except Exception:
            descendant_pids = None
        if descendant_pids is not None:
            if descendant_pids:
                # Native termination avoids taskkill /T delay for known descendants.
                descendants_ok = True
                for descendant_pid in reversed(descendant_pids):
                    if asyncio.get_running_loop().time() >= deadline:
                        descendants_ok = False
                        break
                    if not _windows_terminate_process(descendant_pid, deadline):
                        descendants_ok = False
                leader_ok = _windows_terminate_process(process.pid, deadline)
                return descendants_ok and leader_ok
            descendants_ok = True
        else:
            descendants_ok = True
        try:
            taskkill = subprocess.Popen(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (FileNotFoundError, OSError):
            _kill_codex_process(process)
            return False
        taskkill_wait = asyncio.create_task(asyncio.to_thread(taskkill.wait))
        completed, result = await _wait_task_until(taskkill_wait, deadline)
        if not completed:
            try:
                taskkill.kill()
            except OSError:
                pass
            completed, result = await _cancel_task_until(taskkill_wait, deadline)
        if (
            not completed
            or isinstance(result, BaseException)
            or taskkill.returncode != 0
        ):
            _kill_codex_process(process)
            return False
        return descendants_ok

    if process_group_id is None:
        _kill_codex_process(process)
        return False
    import signal

    try:
        os.killpg(
            process_group_id,
            getattr(signal, "SIGKILL", signal.SIGTERM) if force else signal.SIGTERM,
        )
    except ProcessLookupError:
        return True
    except (OSError, RuntimeError):
        _kill_codex_process(process)
        return False


async def _cleanup_codex_process(
    process: asyncio.subprocess.Process,
    *,
    process_group_id: int | None,
    communication_task: asyncio.Future[Any] | None,
    deadline: float,
) -> _CodexCleanupResult:
    process_wait_task = asyncio.create_task(process.wait())
    tree_terminated = True
    try:
        if os.name == "nt":
            tree_terminated = await _terminate_codex_process_tree(
                process,
                process_group_id=process_group_id,
                force=True,
                deadline=deadline,
            )
        else:
            term_ok = await _terminate_codex_process_tree(
                process,
                process_group_id=process_group_id,
                deadline=deadline,
            )
            grace_deadline = min(
                deadline,
                asyncio.get_running_loop().time() + _CODEX_TERMINATION_GRACE_SECONDS,
            )
            await _wait_task_until(process_wait_task, grace_deadline)
            kill_ok = await _terminate_codex_process_tree(
                process,
                process_group_id=process_group_id,
                force=True,
                deadline=deadline,
            )
            tree_terminated = term_ok and kill_ok
    except Exception:
        tree_terminated = False
        _kill_codex_process(process)

    process_reaped = await _reap_process_until(
        process,
        deadline,
        wait_task=process_wait_task,
    )
    communication_reaped = True
    if communication_task is not None:
        communication_reaped, _ = await _cancel_task_until(communication_task, deadline)
    return _CodexCleanupResult(
        tree_terminated=tree_terminated,
        communication_reaped=communication_reaped,
        process_reaped=process_reaped,
    )


async def _shielded_codex_cleanup(
    process: asyncio.subprocess.Process,
    *,
    process_group_id: int | None,
    communication_task: asyncio.Future[Any] | None,
    deadline: float,
) -> _CodexCleanupResult:
    cleanup_task = asyncio.create_task(
        _cleanup_codex_process(
            process,
            process_group_id=process_group_id,
            communication_task=communication_task,
            deadline=deadline,
        )
    )
    completed, result = await _shielded_wait_task_until(cleanup_task, deadline)
    if completed and isinstance(result, _CodexCleanupResult):
        return result
    await _cancel_task_until(cleanup_task, deadline)
    return _CodexCleanupResult(False, False, False)


def _codex_process_failure_reason(stdout: bytes, stderr: bytes) -> str:
    text = b"\n".join((stdout, stderr)).decode("utf-8", errors="replace").casefold()
    if any(
        marker in text
        for marker in ("429", "rate limit", "rate-limit", "too many requests", "quota")
    ):
        return "rate-limit"
    return "process-exit"


async def _read_codex_stream_bounded(stream: asyncio.StreamReader) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(min(64 * 1024, _MAX_MODEL_RESPONSE_BYTES + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > _MAX_MODEL_RESPONSE_BYTES:
            raise _CodexAttemptError("response-too-large")
        chunks.append(chunk)


async def _communicate_codex_bounded(
    process: asyncio.subprocess.Process,
    prompt: bytes,
) -> tuple[bytes, bytes]:
    stdin = getattr(process, "stdin", None)
    stdout = getattr(process, "stdout", None)
    stderr = getattr(process, "stderr", None)
    if stdin is None or stdout is None or stderr is None:
        stdout_data, stderr_data = await process.communicate(input=prompt)
        if (
            len(stdout_data) > _MAX_MODEL_RESPONSE_BYTES
            or len(stderr_data) > _MAX_MODEL_RESPONSE_BYTES
        ):
            raise _CodexAttemptError("response-too-large")
        return stdout_data, stderr_data

    stdin.write(prompt)
    await stdin.drain()
    stdin.close()
    read_tasks = (
        asyncio.create_task(_read_codex_stream_bounded(stdout)),
        asyncio.create_task(_read_codex_stream_bounded(stderr)),
    )
    try:
        stdout_data, stderr_data = await asyncio.gather(*read_tasks)
        await process.wait()
        return stdout_data, stderr_data
    finally:
        for task in read_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*read_tasks, return_exceptions=True)


def _serialize_codex_messages(
    messages: Sequence[ChatMessage],
    *,
    working_root: Path | None = None,
) -> bytes:
    serialized_messages: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    attachment_index = 0
    for message in messages:
        serialized_message = message.model_dump()
        if message.attachments:
            serialized_attachments: list[dict[str, object]] = []
            for attachment in message.attachments:
                content = _validated_attachment_bytes(attachment)
                suffix = ".png" if attachment.media_type == "image/png" else ".jpg"
                relative_path = PurePosixPath(
                    "evidence",
                    f"{attachment_index:04d}{suffix}",
                )
                attachment_index += 1
                if working_root is not None:
                    isolated_path = working_root.joinpath(*relative_path.parts)
                    isolated_path.parent.mkdir(parents=True, exist_ok=True)
                    isolated_path.write_bytes(content)
                serialized_attachment: dict[str, object] = {
                    **_attachment_audit_metadata(attachment),
                    "path": relative_path.as_posix(),
                }
                serialized_attachments.append(serialized_attachment)
                manifest.append(serialized_attachment)
            serialized_message["attachments"] = serialized_attachments
        serialized_messages.append(serialized_message)

    if not manifest:
        return serialize_transport_json(serialized_messages)
    return serialize_transport_json(
        {
            "messages": serialized_messages,
            "evidence_manifest": manifest,
            "evidence_policy": (
                "Only read evidence files listed in evidence_manifest. "
                "Do not inspect the repository or conversation history."
            ),
        }
    )


def _schema_allows_null(schema: object) -> bool:
    if not isinstance(schema, Mapping):
        return False
    schema_mapping = cast(Mapping[object, object], schema)
    if schema_mapping.get("type") == "null":
        return True
    types = schema_mapping.get("type")
    if isinstance(types, Sequence) and not isinstance(types, (str, bytes)):
        if "null" in types:
            return True
    any_of = schema_mapping.get("anyOf")
    if isinstance(any_of, Sequence) and not isinstance(any_of, (str, bytes)):
        return any(_schema_allows_null(item) for item in cast(Sequence[object], any_of))
    return False


def _codex_transport_schema(schema: type[BaseModel]) -> dict[str, object]:
    """Adapt root optional fields to Codex strict-schema requirements only."""

    payload = cast(
        dict[str, object],
        json.loads(json.dumps(schema.model_json_schema(), ensure_ascii=True)),
    )
    properties_value = payload.get("properties")
    if not isinstance(properties_value, dict):
        return payload
    properties = cast(dict[str, object], properties_value)
    current_required = payload.get("required")
    required_names: set[str] = set()
    if isinstance(current_required, Sequence) and not isinstance(
        current_required, (str, bytes)
    ):
        required_names = {
            name
            for name in cast(Sequence[object], current_required)
            if isinstance(name, str)
        }
    for name, property_schema in tuple(properties.items()):
        if name in required_names or _schema_allows_null(property_schema):
            continue
        properties[name] = {"anyOf": [property_schema, {"type": "null"}]}
    payload["required"] = list(properties)
    return payload


class CodexStructuredClient(_StructuredCallSupport):
    """Codex CLI adapter with local schema validation and bounded retries."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        *,
        call_limiter: asyncio.Semaphore | None = None,
    ) -> None:
        super().__init__(
            settings,
            provider_id="codex-cli",
            provider_version="codex-cli",
        )
        self._call_limiter = call_limiter

    async def complete(
        self,
        schema: type[SchemaT],
        messages: Sequence[ChatMessage | Mapping[str, object]],
        model: str,
        role: ModelRole,
    ) -> SchemaT:
        normalized_messages = _normalize_messages(messages)
        _validate_model_attachments(normalized_messages)
        role_value = ModelRole(role)
        prompt_digest = _prompt_digest(normalized_messages)
        schema_version = _schema_version(schema)
        retry_policy = self.settings.retry_policy
        retries: list[RetryEvent] = []
        attempts = 0
        started = time.perf_counter()
        last_reason = "model call failed"
        response_metadata: dict[str, object] = {"failure": last_reason}
        request_metadata = self._request_metadata(
            role_value,
            model,
            normalized_messages,
            schema_version,
            reasoning_effort=_reasoning_effort(self.settings, role_value),
        )

        while attempts < retry_policy.max_attempts:
            attempts += 1
            request_metadata = self._request_metadata(
                role_value,
                model,
                normalized_messages,
                schema_version,
                reasoning_effort=_reasoning_effort(self.settings, role_value),
            )
            try:
                parsed = await self._run_attempt(
                    schema, normalized_messages, model, role_value
                )
                result = schema.model_validate(parsed)
            except _CodexAttemptError as error:
                last_reason = error.reason
                response_metadata = {"failure": last_reason}
                if attempts >= retry_policy.max_attempts:
                    break
                retries.append(
                    self._retry(
                        role_value,
                        model,
                        attempts,
                        last_reason,
                        None,
                        retry_policy,
                        len(retries) + 1,
                    )
                )
                await self._sleep(retries[-1].delay_seconds)
                continue
            except TransportBudgetError:
                raise
            except (ValueError, TypeError, ValidationError):
                last_reason = "invalid structured output"
                response_metadata = {"failure": "invalid-structured-output"}
                if attempts >= retry_policy.max_attempts:
                    break
                retries.append(
                    self._retry(
                        role_value,
                        model,
                        attempts,
                        "invalid-structured-output",
                        None,
                        retry_policy,
                        len(retries) + 1,
                    )
                )
                await self._sleep(retries[-1].delay_seconds)
                continue

            self._record(
                role_value,
                model,
                prompt_digest,
                schema_version,
                attempts,
                started,
                request_metadata,
                {"status": "success"},
                TokenUsage(),
                retries,
            )
            return result

        self._record(
            role_value,
            model,
            prompt_digest,
            schema_version,
            attempts,
            started,
            request_metadata,
            response_metadata,
            TokenUsage(),
            retries,
        )
        raise ModelFailureError(last_reason)

    @staticmethod
    def _request_metadata(
        role: ModelRole,
        model: str,
        messages: Sequence[ChatMessage],
        schema_version: str,
        reasoning_effort: str | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "role": role.value,
            "model": model,
            "messages": [_audit_message_dump(message) for message in messages],
            "schema_version": schema_version,
        }
        if reasoning_effort is not None:
            metadata["reasoning_effort"] = reasoning_effort
        return metadata

    async def _run_attempt(
        self,
        schema: type[BaseModel],
        messages: Sequence[ChatMessage],
        model: str,
        role: ModelRole,
    ) -> object:
        schema_bytes = serialize_transport_json(
            _codex_transport_schema(schema),
            sort_keys=False,
        )
        output = ""
        loop = asyncio.get_running_loop()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                working_root = Path(temp_dir)
                prompt = _serialize_codex_messages(
                    messages,
                    working_root=working_root,
                )
                if role in _REPORT_ROLES:
                    enforce_transport_size(prompt, schema_bytes)
                schema_path = working_root / "schema.json"
                response_path = working_root / "response.json"
                schema_path.write_bytes(schema_bytes)
                command = [
                    "codex",
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--strict-config",
                    "-c",
                    f'default_permissions="{_CODEX_PERMISSION_PROFILE}"',
                    "-c",
                    _CODEX_PERMISSION_CONFIG,
                ]
                reasoning_effort = _reasoning_effort(self.settings, role)
                if reasoning_effort is not None:
                    command.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])
                command.extend(
                    [
                        "--model",
                        model,
                        "--output-schema",
                        str(schema_path),
                        "--output-last-message",
                        str(response_path),
                        "-",
                    ]
                )
                async with _model_call_slot(self._call_limiter):
                    spawn_task = asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            *command,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=working_root,
                            **_codex_process_options(),
                        )
                    )
                    try:
                        process = await asyncio.shield(spawn_task)
                    except asyncio.CancelledError:
                        cleanup_deadline = loop.time() + _CODEX_CLEANUP_TIMEOUT_SECONDS
                        recovered, spawn_result = await _shielded_wait_task_until(
                            spawn_task, cleanup_deadline
                        )
                        if recovered and not isinstance(spawn_result, BaseException):
                            process = cast(asyncio.subprocess.Process, spawn_result)
                            await _shielded_codex_cleanup(
                                process,
                                process_group_id=_codex_process_group_id(process),
                                communication_task=None,
                                deadline=cleanup_deadline,
                            )
                        else:
                            await _cancel_task_until(spawn_task, cleanup_deadline)
                        raise

                    process_group_id = _codex_process_group_id(process)
                    if role in _REPORT_ROLES:
                        enforce_transport_size(prompt, schema_bytes)
                    communication_task = asyncio.create_task(
                        _communicate_codex_bounded(process, prompt)
                    )
                    stdout_data = b""
                    stderr_data = b""
                    cleanup_attempted = False
                    try:
                        if self.settings.timeout_seconds is None:
                            communication_result = await asyncio.shield(
                                communication_task
                            )
                        else:
                            call_deadline = loop.time() + self.settings.timeout_seconds
                            completed, communication_result = await _wait_task_until(
                                communication_task, call_deadline
                            )
                            if not completed:
                                cleanup = await _shielded_codex_cleanup(
                                    process,
                                    process_group_id=process_group_id,
                                    communication_task=communication_task,
                                    deadline=(
                                        loop.time() + _CODEX_CLEANUP_TIMEOUT_SECONDS
                                    ),
                                )
                                cleanup_attempted = True
                                if not cleanup.success:
                                    raise _CodexAttemptError("process-error")
                                raise _CodexAttemptError("timeout")
                        if isinstance(communication_result, asyncio.CancelledError):
                            raise communication_result
                        if isinstance(communication_result, _CodexAttemptError):
                            raise communication_result
                        if isinstance(communication_result, BaseException):
                            raise _CodexAttemptError("process-error")
                        if not isinstance(communication_result, tuple):
                            raise _CodexAttemptError("process-error")
                        communication_values = cast(
                            tuple[object, ...], communication_result
                        )
                        if (
                            len(communication_values) != 2
                            or not isinstance(communication_values[0], bytes)
                            or not isinstance(communication_values[1], bytes)
                        ):
                            raise _CodexAttemptError("process-error")
                        stdout_data, stderr_data = cast(
                            tuple[bytes, bytes], communication_values
                        )
                    except BaseException:
                        if not cleanup_attempted:
                            await _shielded_codex_cleanup(
                                process,
                                process_group_id=process_group_id,
                                communication_task=communication_task,
                                deadline=(loop.time() + _CODEX_CLEANUP_TIMEOUT_SECONDS),
                            )
                        raise
                    if process.returncode != 0:
                        raise _CodexAttemptError(
                            _codex_process_failure_reason(stdout_data, stderr_data)
                        )
                try:
                    if response_path.stat().st_size > _MAX_MODEL_RESPONSE_BYTES:
                        raise _CodexAttemptError("response-too-large")
                    output = secure_read_bytes(
                        response_path,
                        "Codex structured response",
                        max_bytes=_MAX_MODEL_RESPONSE_BYTES,
                    ).decode("utf-8")
                except UnicodeError:
                    raise ValueError("structured response is not valid UTF-8") from None
                except ValueError:
                    raise _CodexAttemptError("response-too-large") from None
                except OSError:
                    raise _CodexAttemptError("process-error") from None
        except _CodexAttemptError:
            raise
        except OSError:
            raise _CodexAttemptError("process-error") from None

        parsed = json.loads(output)
        if not isinstance(parsed, Mapping):
            raise ValueError("structured response must be one JSON object")
        return cast(dict[str, object], parsed)

    def request_size(
        self,
        schema: type[BaseModel],
        messages: Sequence[ChatMessage],
        *,
        model: str,
        role: ModelRole,
    ) -> int:
        del model
        role_value = ModelRole(role)
        prompt = _serialize_codex_messages(messages)
        schema_bytes = serialize_transport_json(
            _codex_transport_schema(schema),
            sort_keys=False,
        )
        size = transport_size(prompt, schema_bytes)
        if role_value in _REPORT_ROLES:
            enforce_transport_size(prompt, schema_bytes)
        return size


def create_structured_model_client(
    settings: OpenAICompatibleSettings,
    *,
    http_client: httpx.AsyncClient | None = None,
    call_limiter: asyncio.Semaphore | None = None,
) -> StructuredModelClient:
    if settings.mode == "codex":
        return cast(
            StructuredModelClient,
            CodexStructuredClient(settings, call_limiter=call_limiter),
        )
    return cast(
        StructuredModelClient,
        OpenAICompatibleStructuredClient(
            settings,
            http_client=http_client,
            call_limiter=call_limiter,
        ),
    )
