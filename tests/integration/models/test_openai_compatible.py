from __future__ import annotations

import json

import httpx
import pytest

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
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)
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
    assert client.records[0].endpoint_origin == "https://fake-llm.test"
    assert client.records[0].token_usage.total_tokens == 19
    assert client.records[0].attempts == 2
    assert client.manifest(ModelRole.COARSE_SCENT, "scent-model").role == (
        ModelRole.COARSE_SCENT
    )
    assert "secret-api-key" not in json.dumps(client.records[0].request)
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
