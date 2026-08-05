"""OpenAI-compatible structured model adapter.

This module is infrastructure. Domain and application code depend only on the
contracts in ``ux_analyzer.ports.models``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, ValidationError

from ux_analyzer.ports.models import (
    ChatMessage,
    ModelCallRecord,
    ModelManifest,
    ModelRole,
    RetryEvent,
    RetryPolicy,
    StructuredModelClient,
    TokenUsage,
)

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_REDACTED = "[REDACTED]"
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
_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


class ModelConfigurationError(ValueError):
    """Raised when model environment/configuration is incomplete or unsafe."""


class ModelFailureError(RuntimeError):
    """Terminal model failure after classification and bounded retries."""


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
        for secret in sorted(secret_values, key=len, reverse=True):
            sanitized = sanitized.replace(secret, _REDACTED)
        return sanitized
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
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    redaction_values: tuple[str, ...] = ()

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
        for name in ("scent_reasoning_effort", "cognitive_reasoning_effort"):
            object.__setattr__(
                self,
                name,
                _normalize_reasoning_effort(getattr(self, name), name=name),
            )
        if self.timeout_seconds <= 0:
            raise ModelConfigurationError("timeout_seconds must be greater than zero")
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
    ) -> OpenAICompatibleSettings:
        if environ is None:
            load_environment_file(dotenv_path)
            values: Mapping[str, str] = os.environ
        else:
            merged_values = dict(environ)
            load_environment_file(dotenv_path, environ=merged_values)
            values = merged_values
        mode = _normalize_llm_mode(values.get("UXA_LLM_MODE", "api"))
        names = (
            (
                "UXA_LLM_BASE_URL",
                "UXA_LLM_API_KEY",
            )
            if mode == "api"
            else ()
        ) + (
            "UXA_SCENT_MODEL",
            "UXA_COGNITIVE_MODEL",
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
        return cls(
            base_url=base_url,
            api_key=api_key,
            scent_model=values["UXA_SCENT_MODEL"],
            cognitive_model=values["UXA_COGNITIVE_MODEL"],
            mode=mode,
            scent_reasoning_effort=(
                values.get("UXA_LLM_SCENT_REASONING_EFFORT") or None
            ),
            cognitive_reasoning_effort=(
                values.get("UXA_LLM_COGNITIVE_REASONING_EFFORT") or None
            ),
        )

    @classmethod
    def model_validate(cls, value: Mapping[str, object]) -> OpenAICompatibleSettings:
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
        timeout_value = value.get("timeout_seconds", 30.0)
        redaction_value = value.get("redaction_values", ())
        if not isinstance(redaction_value, Sequence) or isinstance(
            redaction_value, (str, bytes)
        ):
            raise ModelConfigurationError("redaction_values must be a sequence")
        mode = _normalize_llm_mode(value.get("mode", "api"))
        if mode == "api":
            base_url = str(value["base_url"])
            api_key = str(value["api_key"])
        else:
            base_url = ""
            api_key = ""
        return cls(
            base_url=base_url,
            api_key=api_key,
            scent_model=str(value["scent_model"]),
            cognitive_model=str(value["cognitive_model"]),
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
            timeout_seconds=_as_float(timeout_value, name="timeout_seconds"),
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
            f"scent_reasoning_effort={self.scent_reasoning_effort!r}, "
            f"cognitive_reasoning_effort={self.cognitive_reasoning_effort!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"retry_policy={self.retry_policy!r})"
        )


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
        normalized.append(ChatMessage(role=role, content=content))
    if not normalized:
        raise ValueError("structured model call needs at least one message")
    return tuple(normalized)


def _prompt_digest(messages: Sequence[ChatMessage]) -> str:
    canonical = json.dumps(
        [message.model_dump() for message in messages],
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


def _response_body(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


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


def _structured_content(body: object) -> object:
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
    content = message_mapping.get("content")
    if isinstance(content, Mapping):
        return cast(dict[str, object], content)
    if not isinstance(content, str):
        raise ValueError("response message has no JSON content")
    parsed = json.loads(content)
    if not isinstance(parsed, Mapping):
        raise ValueError("structured response must be one JSON object")
    return cast(dict[str, object], parsed)


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
    ) -> RetryEvent:
        event = RetryEvent(
            role=role,
            model=model,
            attempt=attempt,
            reason=reason,
            status_code=status_code,
            delay_seconds=policy.delay_for_retry(retry_number),
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
    ) -> None:
        super().__init__(
            settings,
            provider_id="openai-compatible-structured",
            provider_version="openai-compatible-v1",
        )
        self._http_client = http_client or httpx.AsyncClient(
            timeout=settings.timeout_seconds
        )

    async def complete(
        self,
        schema: type[SchemaT],
        messages: Sequence[ChatMessage | Mapping[str, object]],
        model: str,
        role: ModelRole,
    ) -> SchemaT:
        normalized_messages = _normalize_messages(messages)
        role_value = ModelRole(role)
        prompt_digest = _prompt_digest(normalized_messages)
        schema_version = _schema_version(schema)
        retry_policy = self.settings.retry_policy
        retries: list[RetryEvent] = []
        attempts = 0
        started = time.perf_counter()
        response_payload: object = {}
        # This endpoint hangs on cognitive discriminated schemas instead of
        # returning a useful unsupported-schema response. Keep local validation
        # while using its compatible JSON-object mode for that role.
        mode = "json-object" if role_value is ModelRole.COGNITIVE else "strict"
        last_reason = "model call failed"

        while attempts < retry_policy.max_attempts:
            attempts += 1
            request_payload = self._request_payload(
                schema, normalized_messages, model, role_value, mode
            )
            try:
                response = await self._http_client.post(
                    f"{self.settings.base_url}/chat/completions",
                    headers={
                        "accept": "application/json",
                        "content-type": "application/json",
                        "authorization": f"Bearer {self.settings.api_key}",
                    },
                    json=request_payload,
                )
                response_payload = _response_body(response)
            except httpx.TransportError as error:
                last_reason = _transport_error_category(error)
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
                        )
                    )
                    await self._sleep(retries[-1].delay_seconds)
                    continue
                break
            if 500 <= response.status_code <= 599:
                last_reason = "server error"
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
                last_reason = "request rejected"
                break

            try:
                parsed = _structured_content(response_payload)
                result = schema.model_validate(parsed)
            except (ValueError, TypeError, ValidationError):
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
                request_payload,
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
            schema, normalized_messages, model, role_value, mode
        )
        self._record(
            role_value,
            model,
            prompt_digest,
            schema_version,
            attempts,
            started,
            request_payload,
            (
                cast(Mapping[str, object], response_payload)
                if isinstance(response_payload, Mapping)
                else {"error": response_payload}
            ),
            (
                _usage(cast(Mapping[str, object], response_payload))
                if isinstance(response_payload, Mapping)
                else TokenUsage()
            ),
            retries,
        )
        raise ModelFailureError(last_reason)

    def _request_payload(
        self,
        schema: type[BaseModel],
        messages: Sequence[ChatMessage],
        model: str,
        role: ModelRole,
        mode: str,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
        }
        reasoning_effort = (
            self.settings.cognitive_reasoning_effort
            if role is ModelRole.COGNITIVE
            else self.settings.scent_reasoning_effort
        )
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


class _CodexAttemptError(RuntimeError):
    """Sanitized failure category from one Codex subprocess attempt."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _serialize_codex_messages(messages: Sequence[ChatMessage]) -> bytes:
    return json.dumps(
        [message.model_dump() for message in messages],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


class CodexStructuredClient(_StructuredCallSupport):
    """Codex CLI adapter with local schema validation and bounded retries."""

    def __init__(self, settings: OpenAICompatibleSettings) -> None:
        super().__init__(
            settings,
            provider_id="codex-cli",
            provider_version="codex-cli",
        )

    async def complete(
        self,
        schema: type[SchemaT],
        messages: Sequence[ChatMessage | Mapping[str, object]],
        model: str,
        role: ModelRole,
    ) -> SchemaT:
        normalized_messages = _normalize_messages(messages)
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
            role_value, model, normalized_messages, schema_version
        )

        while attempts < retry_policy.max_attempts:
            attempts += 1
            request_metadata = self._request_metadata(
                role_value, model, normalized_messages, schema_version
            )
            try:
                parsed = await self._run_attempt(schema, normalized_messages, model)
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
    ) -> dict[str, object]:
        return {
            "role": role.value,
            "model": model,
            "messages": [message.model_dump() for message in messages],
            "schema_version": schema_version,
        }

    async def _run_attempt(
        self,
        schema: type[BaseModel],
        messages: Sequence[ChatMessage],
        model: str,
    ) -> object:
        prompt = _serialize_codex_messages(messages)
        output = ""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                schema_path = Path(temp_dir) / "schema.json"
                response_path = Path(temp_dir) / "response.json"
                schema_path.write_text(
                    json.dumps(schema.model_json_schema(), ensure_ascii=True),
                    encoding="utf-8",
                )
                process = await asyncio.create_subprocess_exec(
                    "codex",
                    "exec",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--model",
                    model,
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(response_path),
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    await asyncio.wait_for(
                        process.communicate(input=prompt),
                        timeout=self.settings.timeout_seconds,
                    )
                except TimeoutError as error:
                    try:
                        process.kill()
                        await process.communicate()
                    except OSError:
                        pass
                    raise _CodexAttemptError("timeout") from error
                if process.returncode != 0:
                    raise _CodexAttemptError("process-exit")
                try:
                    output = response_path.read_text(encoding="utf-8")
                except OSError as error:
                    raise _CodexAttemptError("process-error") from error
        except _CodexAttemptError:
            raise
        except OSError as error:
            raise _CodexAttemptError("process-error") from error

        parsed = json.loads(output)
        if not isinstance(parsed, Mapping):
            raise ValueError("structured response must be one JSON object")
        return cast(dict[str, object], parsed)


def create_structured_model_client(
    settings: OpenAICompatibleSettings,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> StructuredModelClient:
    if settings.mode == "codex":
        return cast(StructuredModelClient, CodexStructuredClient(settings))
    return cast(
        StructuredModelClient,
        OpenAICompatibleStructuredClient(settings, http_client=http_client),
    )
