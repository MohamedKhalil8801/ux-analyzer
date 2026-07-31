from __future__ import annotations

import asyncio
import json
import random
import socket
import threading
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

import ux_analyzer.cli as cli
from fixture_app.app import app as fixture_app
from tests.e2e.test_private_data_leakage import _assert_no_leaks, _snapshot
from ux_analyzer.adapters.openai import OpenAICompatibleSettings
from ux_analyzer.application.evaluation import evaluate_experiment_results
from ux_analyzer.application.experiment import ExperimentContext, expand_experiment
from ux_analyzer.config.loader import load_project
from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.benchmark import Budget, ExperimentPolicy
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.run import ObservationRecorded
from ux_analyzer.providers.attention_policy import (
    AttentionPolicyConfig,
    ProgressiveAttentionPolicy,
)
from ux_analyzer.providers.prominence import HeuristicProminenceProvider

DEMO_PROJECT = Path(__file__).parents[2] / "benchmarks" / "demo" / "project.yaml"
CI_SEED = 7


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _run_id(element_id: str) -> str:
    return element_id.split("-viewport-", maxsplit=1)[0]


def _response(content: dict[str, object]) -> JSONResponse:
    return JSONResponse(
        {
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )


def _model_app(
    request_log: list[dict[str, object]], versions: dict[str, str]
) -> FastAPI:
    model_app = FastAPI()
    typed_runs: set[str] = set()
    opened_menu_runs: set[str] = set()

    @model_app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, payload: dict[str, object] = Body(...)
    ) -> JSONResponse:
        assert request.headers.get("authorization") == "Bearer ci-api-key"
        request_log.append(payload)
        messages = payload.get("messages")
        assert isinstance(messages, list) and messages
        user_message = messages[-1]
        assert isinstance(user_message, dict)
        user_payload = json.loads(str(user_message["content"]))
        model = payload.get("model")
        if model == "scent-model":
            elements = user_payload.get("elements", [])
            assert isinstance(elements, list)
            if elements:
                run_id = _run_id(str(elements[0]["element_id"]))
                labels = {str(item["label"]).lower() for item in elements}
                if "defective fixture" in labels:
                    versions[run_id] = "defective"
                elif "improved fixture" in labels:
                    versions[run_id] = "improved"
            return _response(
                {
                    "scores": [
                        {
                            "element_id": item["element_id"],
                            "score": _scent_score(
                                str(item["label"]),
                                versions.get(_run_id(str(item["element_id"])), ""),
                                bool(item["actionable"]),
                            ),
                        }
                        for item in elements
                    ]
                }
            )

        assert model == "cognitive-model"
        newly = user_payload.get("newly_revealed_elements", [])
        remembered = user_payload.get("remembered_elements", [])
        assert isinstance(newly, list) and isinstance(remembered, list)
        elements = [*newly, *remembered]
        run_id = _run_id(str(elements[0]["element_id"])) if elements else "unknown"
        labels = {str(item["label"]).lower() for item in elements}
        if "defective fixture" in labels:
            versions[run_id] = "defective"
        elif "improved fixture" in labels:
            versions[run_id] = "improved"
        action = _cognitive_action(
            str(user_payload["goal"]),
            elements,
            newly,
            run_id,
            versions.get(run_id, ""),
            typed_runs,
            opened_menu_runs,
        )
        return _response({"action": action, "reason": "Deterministic CI fixture path."})

    return model_app


def _scent_score(label: str, version: str, actionable: bool) -> float:
    lowered = label.lower()
    if not actionable:
        return 0.05
    if lowered == "share":
        return 1.0
    if "send invitation" in lowered:
        return 0.6 if version == "defective" else 0.9
    if "invite teammate" in lowered or "members" in lowered:
        return 0.9
    if "email address" in lowered:
        return 0.85
    if "enable protection" in lowered:
        return 0.2
    if "protection" in lowered:
        return 0.95
    if "two-factor authentication" in lowered or "setup code" in lowered:
        return 0.95
    if "verification code" in lowered:
        return 0.75
    if "account menu" in lowered or lowered == "nk":
        return 0.7
    return 0.1


def _cognitive_action(
    goal: str,
    elements: list[dict[str, object]],
    newly: list[dict[str, object]],
    run_id: str,
    version: str,
    typed_runs: set[str],
    opened_menu_runs: set[str],
) -> dict[str, object]:
    actionable = [item for item in elements if item.get("actionable")]

    def find(*labels: str) -> dict[str, object] | None:
        return next(
            (
                item
                for item in actionable
                if any(label in str(item["label"]).lower() for label in labels)
            ),
            None,
        )

    if "invite" in goal.lower():
        input_element = find("email address")
        if input_element is not None and run_id not in typed_runs:
            typed_runs.add(run_id)
            return {
                "kind": "type-fixture",
                "element_id": input_element["element_id"],
                "fixture_key": "invite_email",
            }
        if run_id in typed_runs and (submit := find("send invitation")) is not None:
            return {"kind": "interact", "element_id": submit["element_id"]}
        if (target := find("invite teammate")) is not None:
            return {"kind": "interact", "element_id": target["element_id"]}
        if (
            version == "defective"
            and run_id not in opened_menu_runs
            and (menu := find("open account menu", "nk")) is not None
        ):
            opened_menu_runs.add(run_id)
            return {"kind": "interact", "element_id": menu["element_id"]}
        for label in ("members",):
            if (target := find(label)) is not None:
                return {"kind": "interact", "element_id": target["element_id"]}
    else:
        input_element = find("setup code", "verification code")
        if input_element is not None and run_id not in typed_runs:
            typed_runs.add(run_id)
            return {
                "kind": "type-fixture",
                "element_id": input_element["element_id"],
                "fixture_key": "totp_code",
            }
        if (
            run_id in typed_runs
            and (
                submit := find("enable two-factor authentication", "enable protection")
            )
            is not None
        ):
            return {"kind": "interact", "element_id": submit["element_id"]}
        if (target := find("protection")) is not None:
            return {"kind": "interact", "element_id": target["element_id"]}

    inspectable = newly or elements
    if inspectable:
        return {"kind": "inspect", "element_id": inspectable[0]["element_id"]}
    return {"kind": "wait"}


@pytest_asyncio.fixture
async def production_servers() -> tuple[
    str, str, list[dict[str, object]], dict[str, str]
]:
    fixture_port = _free_port()
    model_port = _free_port()
    request_log: list[dict[str, object]] = []
    versions: dict[str, str] = {}
    fixture_server = uvicorn.Server(
        uvicorn.Config(
            fixture_app, host="127.0.0.1", port=fixture_port, log_level="error"
        )
    )
    model_server = uvicorn.Server(
        uvicorn.Config(
            _model_app(request_log, versions),
            host="127.0.0.1",
            port=model_port,
            log_level="error",
        )
    )
    fixture_thread = threading.Thread(target=fixture_server.run, daemon=True)
    model_thread = threading.Thread(target=model_server.run, daemon=True)
    fixture_thread.start()
    model_thread.start()
    fixture_origin = f"http://127.0.0.1:{fixture_port}"
    model_base_url = f"http://127.0.0.1:{model_port}/v1"
    async with httpx.AsyncClient() as client:
        for url in (
            f"{fixture_origin}/app/ready/improved",
            f"{model_base_url}/chat/completions",
        ):
            for _ in range(200):
                try:
                    response = await client.get(url)
                    if response.status_code < 500:
                        break
                except httpx.ConnectError:
                    await asyncio.sleep(0.01)
            else:
                raise RuntimeError(f"server did not start: {url}")
    yield fixture_origin, model_base_url, request_log, versions
    fixture_server.should_exit = True
    model_server.should_exit = True
    await asyncio.to_thread(fixture_thread.join, 10)
    await asyncio.to_thread(model_thread.join, 10)
    assert not fixture_thread.is_alive()
    assert not model_thread.is_alive()


def _matrix(scenario_id: str) -> cli._ResolvedMatrix:
    loaded = load_project(DEMO_PROJECT)
    project = replace(
        loaded.project,
        scenarios=tuple(
            replace(
                scenario,
                budget=replace(scenario.budget, timeout_seconds=300),
            )
            for scenario in loaded.project.scenarios
        ),
    )
    loaded = replace(
        loaded,
        project=project,
        runtime=replace(
            loaded.runtime,
            attention=replace(loaded.runtime.attention, coarse_scent_weight=8.0),
        ),
    )
    definition = next(item for item in project.experiments if item.id == "core-pair")
    reduced = replace(definition, seeds=(CI_SEED,), run_count=1)
    specs = expand_experiment(ExperimentContext(reduced, project, loaded.config_digest))
    specs = tuple(spec for spec in specs if spec.scenario.id == scenario_id)
    return cli._ResolvedMatrix(loaded=loaded, definition=reduced, specs=specs)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", ("invite-teammate", "enable-2fa"))
async def test_production_cli_composition_runs_real_core_matrix_and_findings(
    scenario_id: str,
    production_servers: tuple[str, str, list[dict[str, object]], dict[str, str]],
    tmp_path: Path,
) -> None:
    fixture_origin, model_base_url, requests, versions = production_servers
    matrix = _matrix(scenario_id)
    versions.update(
        {spec.run_id: spec.application_version.kind.value for spec in matrix.specs}
    )
    output = tmp_path / scenario_id
    settings = OpenAICompatibleSettings.model_validate(
        {
            "base_url": model_base_url,
            "api_key": "ci-api-key",
            "scent_model": "scent-model",
            "cognitive_model": "cognitive-model",
            "timeout_seconds": 300,
            "retry_policy": {"max_attempts": 1, "base_delay_seconds": 0},
        }
    )

    result = await cli._execute_matrix(
        matrix,
        output=output,
        workers=2,
        fixture_origin=fixture_origin,
        settings=cli._settings_with_fixture_redaction(settings, matrix.loaded),
    )

    assert result.failures == (), [failure.message for failure in result.failures]
    assert len(result.results) == 8
    assert {
        (
            spec.scenario.id,
            spec.application_version.kind.value,
            spec.persona.id,
            spec.policy,
        )
        for spec in result.specs
    } == {
        (scenario, version, persona, policy)
        for scenario in (scenario_id,)
        for version in ("defective", "improved")
        for persona in ("first-time-nontechnical", "impatient")
        for policy in (
            ExperimentPolicy.FULL_LIST,
            ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT,
        )
    }
    assert requests

    defective_categories: set[str] = set()
    improved_categories: set[str] = set()
    for spec, run_result in zip(result.specs, result.results, strict=True):
        assert run_result.outcome.kind == "verified-success", (
            spec.scenario.id,
            spec.application_version.kind.value,
            spec.persona.id,
            spec.policy.value,
            run_result.terminal_reason,
        )
        assert run_result.verification.verified
        assert run_result.metrics is not None
        assert run_result.bundle_path is not None
        assert run_result.state.snapshots
        assert all(
            snapshot.provider_id == "playwright-web"
            for snapshot in run_result.state.snapshots
        )
        assert all(
            finding.evidence_class is not EvidenceClass.UNSUPPORTED_HUMAN_CLAIM
            for finding in run_result.findings
        )
        categories = {finding.category for finding in run_result.findings}
        if spec.application_version.kind.value == "defective":
            defective_categories.update(categories)
        else:
            improved_categories.update(categories)
        bundle = Path(run_result.bundle_path)
        assert (bundle / "manifest.json").is_file()
        assert (bundle / "timeline.jsonl").is_file()
        assert (bundle / "result.json").is_file()
        assert (bundle / "checksums.sha256").is_file()
        assert run_result.evidence.screenshot_artifacts
        assert all(
            (bundle / screenshot.path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            for screenshot in run_result.evidence.screenshot_artifacts
        )
        for event in run_result.state.events:
            if isinstance(event, ObservationRecorded):
                _assert_no_leaks(
                    f"{spec.run_id}:observation",
                    asdict(event.observation),
                    {
                        "api-key": "ci-api-key",
                        "private-control-path": "/__control/",
                        "sensitive-fixture-value": next(
                            iter(spec.scenario.fixture_inputs.values.values())
                        ),
                    },
                )

    assert "unexpected-hierarchy" in defective_categories
    assert "ambiguous-icon-label" in defective_categories
    if scenario_id == "invite-teammate":
        assert "strong-misleading-alternative" in defective_categories
    else:
        assert "missing-feedback" in defective_categories
        assert "missing-feedback" not in improved_categories

    evaluation = evaluate_experiment_results(result.results)
    assert len(evaluation.variant_comparisons) == 4
    summary_path, report_path = cli._complete_experiment(
        result, output=output, runtime=matrix.loaded.runtime
    )
    assert all(item.gate.passed for item in evaluation.variant_comparisons), [
        (
            item.baseline.persona_id,
            item.baseline.policy,
            item.gate.reasons,
            item.gate.baseline_discovery_cost_median,
            item.gate.improved_discovery_cost_median,
        )
        for item in evaluation.variant_comparisons
        if not item.gate.passed
    ]
    assert summary_path.is_file()
    assert report_path is not None and report_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["findings"]
    assert (
        "simulated benchmark evidence"
        in report_path.read_text(encoding="utf-8").lower()
    )


@pytest.mark.e2e
def test_seeded_progressive_attention_has_repeatable_but_varying_paths() -> None:
    snapshot = _snapshot()
    state = AttentionState.initial(Budget(10, 10, 5, 10), confidence=0.5, frustration=0)
    scores = HeuristicProminenceProvider().score(snapshot)
    policy = ProgressiveAttentionPolicy(AttentionPolicyConfig(batch_size=1))
    first = policy.next_observation(state, snapshot, scores, (), random.Random(7))
    repeat = policy.next_observation(state, snapshot, scores, (), random.Random(7))
    paths = {
        policy.next_observation(
            state, snapshot, scores, (), random.Random(seed)
        ).selected_ids
        for seed in range(1, 20)
    }

    assert first.selected_ids == repeat.selected_ids
    assert len(paths) > 1
