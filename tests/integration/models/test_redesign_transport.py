"""Transport smoke tests for redesign roles over both transports (Task 4).

The same prompt/contract must flow through the OpenAI-compatible HTTP
transport (tool-call mode, image segments as attachments) and the Codex
subprocess transport (schema file, evidence files). Codex coverage here is
stubbed at the process boundary; live behavior stays best-effort per ADR 0007.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path

import httpx
import pytest

from ux_analyzer.adapters.openai import (
    CodexStructuredClient,
    ModelFailureError,
    OpenAICompatibleSettings,
    OpenAICompatibleStructuredClient,
)
from ux_analyzer.ports.models import ModelAttachment, ModelRole
from ux_analyzer.providers.redesign import (
    RedesignCriticMerger,
    RedesignProposer,
)


def _settings(**overrides: object) -> OpenAICompatibleSettings:
    values: dict[str, object] = {
        "base_url": "https://fake-llm.test/v1",
        "api_key": "secret-api-key",
        "scent_model": "scent-model",
        "cognitive_model": "cognitive-model",
        "retry_policy": {"max_attempts": 1, "base_delay_seconds": 0},
    }
    values.update(overrides)
    return OpenAICompatibleSettings.model_validate(values)


def _jpeg_attachment(tmp_path: Path) -> ModelAttachment:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(90, 90, 200)).save(buffer, format="JPEG")
    content = buffer.getvalue()
    path = tmp_path / "segment-000.jpg"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return ModelAttachment(
        evidence_id=f"capture:segment:{digest[:16]}",
        path=path,
        media_type="image/jpeg",
        sha256=digest,
    )


_PROPOSER_OUTPUT = {
    "page_understanding": {
        "page_url": "https://fixture.test/",
        "intent": "Explain the product and drive signups.",
        "audience_inference": "Likely first-time visitors comparing plans.",
        "section_relationships": "Hero feeds a feature row and a pricing block.",
    },
    "proposals": [
        {
            "proposal_id": "p1",
            "page_url": "https://fixture.test/",
            "category": "whitespace",
            "title": "Widen spacing between pricing cards",
            "observation": "Cards sit 8px apart and read as one block.",
            "rationale": "Grouping clarity suffers without separation.",
            "change": "Raise the gap to 32px.",
            "principle_ids": ["gestalt-proximity"],
            "impact": "medium",
            "effort": "small",
            "section_refs": [
                {
                    "url": "https://fixture.test/",
                    "section_label": "Pricing cards",
                    "box": {"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
                    "summary": "Three pricing cards in a row.",
                }
            ],
            "also_affects": [],
            "deliberate_choice_check": None,
        }
    ],
}

_CRITIC_OUTPUT = {
    "final_proposals": [
        {**_PROPOSER_OUTPUT["proposals"][0], "proposal_id": "m1"}  # type: ignore[dict-item]
    ],
    "killed": [{"proposal_id": "p2", "reason": "duplicates m1"}],
    "consistency_notes": ["Same card gap rule applied on every page."],
}


@pytest.mark.asyncio
async def test_proposer_uses_tool_call_transport_with_image_attachment(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "uxa_redesign_proposer",
                                        "arguments": json.dumps(_PROPOSER_OUTPUT),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)
    proposer = RedesignProposer(client, model="redesign-model")
    attachment = _jpeg_attachment(tmp_path)

    result = await proposer.analyze(
        {"url": "https://fixture.test/", "title": "Fixture"},
        audience="",
        principles=[{"id": "gestalt-proximity", "name": "Proximity"}],
        attachments=(attachment,),
    )

    assert result.proposals[0].proposal_id == "p1"
    request = requests[0]
    assert request["tool_choice"] == {
        "type": "function",
        "function": {"name": "uxa_redesign_proposer"},
    }
    user_message = request["messages"][-1]
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][1]["type"] == "image_url"
    assert user_message["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_critic_uses_tool_call_transport_and_validates_locally() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "uxa_redesign_critic_merger",
                                        "arguments": json.dumps(_CRITIC_OUTPUT),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(_settings(), http_client=http_client)
    critic = RedesignCriticMerger(client, model="redesign-model")

    result = await critic.review(
        [{"proposal_id": "p1", "page_url": "https://fixture.test/"}],
        [{"url": "https://fixture.test/", "digest": "abc"}],
        audience="",
        principles=[],
    )

    assert result.final_proposals[0].proposal_id == "m1"
    assert result.killed[0].proposal_id == "p2"
    assert requests[0]["tools"][0]["function"]["name"] == "uxa_redesign_critic_merger"  # type: ignore[index]
    await http_client.aclose()


def test_model_for_role_prefers_redesign_model_with_report_fallback() -> None:
    explicit = _settings(
        report_model="report-model", redesign_model="redesign-model"
    )
    fallback = _settings(report_model="report-model")
    assert explicit.model_for_role(ModelRole.REDESIGN_PROPOSER) == "redesign-model"
    assert explicit.model_for_role(ModelRole.REDESIGN_CRITIC_MERGER) == (
        "redesign-model"
    )
    assert fallback.model_for_role(ModelRole.REDESIGN_PROPOSER) == "report-model"
    neither = _settings()
    with pytest.raises(Exception, match="redesign_model or report_model"):
        neither.model_for_role(ModelRole.REDESIGN_PROPOSER)


def test_from_env_reads_redesign_model_with_report_fallback() -> None:
    explicit = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_BASE_URL": "https://fake-llm.test/v1",
            "UXA_LLM_API_KEY": "key",
            "UXA_SCENT_MODEL": "scent",
            "UXA_COGNITIVE_MODEL": "cognitive",
            "UXA_REPORT_MODEL": "report",
            "UXA_REDESIGN_MODEL": "redesign",
        },
        report_synthesis_enabled=True,
    )
    assert explicit.redesign_model == "redesign"
    fallback = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_BASE_URL": "https://fake-llm.test/v1",
            "UXA_LLM_API_KEY": "key",
            "UXA_SCENT_MODEL": "scent",
            "UXA_COGNITIVE_MODEL": "cognitive",
            "UXA_REPORT_MODEL": "report",
        },
        report_synthesis_enabled=True,
    )
    assert fallback.redesign_model == "report"


@pytest.mark.asyncio
async def test_proposer_degrades_to_strict_then_json_object_on_invalid_output() -> None:
    """A provider that ignores the function-calling transport must not exhaust
    every attempt in tool-call mode: redesign roles degrade tool-call →
    strict → json-object, mirroring the report-role ladder (ADR 0007)."""

    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json"}}]},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(retry_policy={"max_attempts": 3, "base_delay_seconds": 0}),
        http_client=http_client,
    )
    proposer = RedesignProposer(client, model="redesign-model")

    with pytest.raises(ModelFailureError, match="invalid structured output"):
        await proposer.analyze(
            {"url": "https://fixture.test/", "title": "Fixture"},
            audience="",
            principles=[],
        )

    assert _transport_requests(requests) == [
        "tool-call",
        "json_schema",
        "json_object",
    ]
    reanchors = [
        message
        for record in requests
        for message in record["messages"]  # type: ignore[union-attr]
        if "Return exactly one valid JSON object" in str(message.get("content"))
    ]
    assert len(reanchors) == 2
    failure_record = client.records[0]
    assert failure_record.response["failure"] == "invalid structured output"
    diagnostics = failure_record.response["diagnostics"]
    assert diagnostics["attempt_count"] == 3
    assert diagnostics["response_mode"] == "json-object"
    await http_client.aclose()


def _transport_requests(requests: list[dict[str, object]]) -> list[str]:
    """Return the response mode of each request in order."""

    modes: list[str] = []
    for request in requests:
        response_format = request.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type"):
            modes.append(str(response_format.get("type")))
        else:
            modes.append("tool-call")
    return modes


@pytest.mark.asyncio
async def test_proposer_recovers_fenced_json_in_degraded_mode() -> None:
    """Degraded redesign transports share the report-role recovery parser:
    a fenced JSON object is accepted and schema-validated locally."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "```json\nnope\n```"}}]},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                + json.dumps(_PROPOSER_OUTPUT)
                                + "\n```"
                            )
                        }
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleStructuredClient(
        _settings(retry_policy={"max_attempts": 3, "base_delay_seconds": 0}),
        http_client=http_client,
    )
    proposer = RedesignProposer(client, model="redesign-model")

    result = await proposer.analyze(
        {"url": "https://fixture.test/", "title": "Fixture"},
        audience="",
        principles=[],
    )

    assert result.proposals[0].proposal_id == "p1"
    await http_client.aclose()


class _FakeCodexProcess:
    """Mirrors the existing codex test double: no stream attributes so the
    adapter falls back to ``communicate(input=prompt)``."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        del input
        return (b"codex stdout must not be recorded", b"")

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _patch_codex_output(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output: str,
    schemas: list[object],
) -> None:
    async def create_subprocess_exec(
        *args: object, **kwargs: object
    ) -> _FakeCodexProcess:
        del kwargs
        args_tuple = tuple(args)
        schema_path = Path(args_tuple[args_tuple.index("--output-schema") + 1])
        schemas.append(json.loads(schema_path.read_text(encoding="utf-8")))
        output_path = Path(args_tuple[args_tuple.index("--output-last-message") + 1])
        output_path.write_text(output, encoding="utf-8")
        return _FakeCodexProcess(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)


@pytest.mark.asyncio
async def test_codex_transport_runs_proposer_with_same_schema_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schemas: list[object] = []
    _patch_codex_output(
        monkeypatch,
        output=json.dumps(_PROPOSER_OUTPUT),
        schemas=schemas,
    )
    client = CodexStructuredClient(_settings(mode="codex"))
    proposer = RedesignProposer(client, model="redesign-model")

    result = await proposer.analyze(
        {"url": "https://fixture.test/", "title": "Fixture"},
        audience="",
        principles=[],
    )

    assert result.proposals[0].proposal_id == "p1"
    assert schemas, "codex transport must write the response schema file"
    schema_payload = schemas[0]
    properties = schema_payload.get("properties", {})  # type: ignore[union-attr]
    assert "page_understanding" in properties  # type: ignore[operator]
    assert "proposals" in properties  # type: ignore[operator]


@pytest.mark.asyncio
async def test_codex_transport_runs_critic_with_same_schema_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schemas: list[object] = []
    _patch_codex_output(
        monkeypatch,
        output=json.dumps(_CRITIC_OUTPUT),
        schemas=schemas,
    )
    client = CodexStructuredClient(_settings(mode="codex"))
    critic = RedesignCriticMerger(client, model="redesign-model")

    result = await critic.review(
        [{"proposal_id": "p1", "page_url": "https://fixture.test/"}],
        [{"url": "https://fixture.test/", "digest": "abc"}],
        audience="",
        principles=[],
    )

    assert result.final_proposals[0].proposal_id == "m1"
    properties = schemas[0].get("properties", {})  # type: ignore[union-attr]
    assert "final_proposals" in properties  # type: ignore[operator]
    assert "killed" in properties  # type: ignore[operator]
