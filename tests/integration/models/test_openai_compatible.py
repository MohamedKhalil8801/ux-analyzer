from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from ux_analyzer.adapters.openai import (
    ModelFailureError,
    OpenAICompatibleSettings,
    OpenAICompatibleStructuredClient,
)
from ux_analyzer.ports.models import ChatMessage, ModelRole
from ux_analyzer.providers.scent import CoarseScentResponse


def _settings(**overrides: object) -> OpenAICompatibleSettings:
    values: dict[str, object] = {
        "base_url": "https://fake-llm.test/v1",
        "api_key": "secret-api-key",
        "scent_model": "scent-model",
        "cognitive_model": "cognitive-model",
        "retry_policy": {"max_attempts": 3, "base_delay_seconds": 0},
    }
    values.update(overrides)
    return OpenAICompatibleSettings.model_validate(values)


@pytest.mark.asyncio
async def test_strict_schema_fallback_validates_locally_and_records_usage() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format unsupported"}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"scores": [{"element_id": "target", "score": 0.7}]}
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(scent_reasoning_effort="low"),
        http_client=http_client,
    )
    result = await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content='{"goal":"Find invite"}'),),
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.scores[0].score == pytest.approx(0.7)
    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"]["type"] == "json_object"
    assert requests[0]["reasoning_effort"] == "low"
    assert requests[1]["reasoning_effort"] == "low"
    assert client.records[0].endpoint_origin == "https://fake-llm.test"
    assert client.records[0].token_usage.total_tokens == 19
    assert client.records[0].attempts == 2
    assert client.manifest(ModelRole.COARSE_SCENT, "scent-model").role == (
        ModelRole.COARSE_SCENT
    )
    assert "secret-api-key" not in json.dumps(client.records[0].request)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_invalid_request_falls_back_from_strict_schema() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "The request could not be processed.",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)

    class SimpleResponse(BaseModel):
        ok: bool

    result = await client.complete(
        SimpleResponse,
        (ChatMessage(role="user", content="{}"),),
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.ok
    assert [request["response_format"]["type"] for request in requests] == [
        "json_schema",
        "json_object",
    ]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_cognitive_role_starts_in_json_object_mode() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)

    class SimpleResponse(BaseModel):
        ok: bool

    result = await client.complete(
        SimpleResponse,
        (ChatMessage(role="user", content="{}"),),
        model="cognitive-model",
        role=ModelRole.COGNITIVE,
    )

    assert result.ok
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in requests[0]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_cognitive_reasoning_effort_is_forwarded_when_configured() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(scent_reasoning_effort="low", cognitive_reasoning_effort="high"),
        http_client=http_client,
    )

    class SimpleResponse(BaseModel):
        ok: bool

    result = await client.complete(
        SimpleResponse,
        (ChatMessage(role="user", content="{}"),),
        model="cognitive-model",
        role=ModelRole.COGNITIVE,
    )

    assert result.ok
    assert requests[0]["reasoning_effort"] == "high"
    await http_client.aclose()


@pytest.mark.asyncio
async def test_invalid_structured_output_retries_only_within_bound() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = "not json" if calls < 3 else json.dumps({"scores": []})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)
    result = await client.complete(
        CoarseScentResponse,
        (ChatMessage(role="user", content="{}"),),
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert result.scores == []
    assert calls == 3
    assert len(client.retry_events) == 2
    assert all(
        event.reason == "invalid-structured-output" for event in client.retry_events
    )
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (httpx.ConnectTimeout("connect timed out"), "connect-timeout"),
        (httpx.ReadTimeout("read timed out"), "read-timeout"),
        (httpx.ConnectError("connection failed"), "connect-error"),
        (httpx.RemoteProtocolError("server disconnected"), "protocol-error"),
    ),
)
async def test_transport_failures_record_safe_specific_category(
    error: httpx.TransportError,
    reason: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise error

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(retry_policy={"max_attempts": 1, "base_delay_seconds": 0}),
        http_client=http_client,
    )

    with pytest.raises(ModelFailureError, match=reason):
        await client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="{}"),),
            model="scent-model",
            role=ModelRole.COARSE_SCENT,
        )

    assert client.records[0].attempts == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_authentication_failure_is_terminal_without_retry() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)

    with pytest.raises(ModelFailureError, match="authentication failure"):
        await client.complete(
            CoarseScentResponse,
            (ChatMessage(role="user", content="{}"),),
            model="scent-model",
            role=ModelRole.COARSE_SCENT,
        )

    assert calls == 1
    assert client.retry_events == ()
    await http_client.aclose()


@pytest.mark.asyncio
async def test_retry_backoff_restarts_for_each_logical_model_call() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls in {1, 3}:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"scores": []})}}]},
        )

    settings = _settings(
        retry_policy={
            "max_attempts": 2,
            "base_delay_seconds": 0.25,
            "max_delay_seconds": 1.0,
            "multiplier": 2.0,
        }
    )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(settings, http_client=http_client)
    delays: list[float] = []

    async def record_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    client._sleep = record_sleep  # type: ignore[method-assign]
    messages = (ChatMessage(role="user", content="{}"),)

    await client.complete(
        CoarseScentResponse,
        messages,
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )
    await client.complete(
        CoarseScentResponse,
        messages,
        model="scent-model",
        role=ModelRole.COARSE_SCENT,
    )

    assert delays == [0.25, 0.25]
    await http_client.aclose()
