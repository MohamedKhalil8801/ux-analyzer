from __future__ import annotations

import asyncio
import json
import random
import socket
import threading
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import uvicorn
import yaml
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from playwright.async_api import Route, async_playwright
from typer.testing import CliRunner

import ux_analyzer.cli as cli
from fixture_app.app import app as fixture_app
from tests.e2e.test_private_data_leakage import _snapshot
from ux_analyzer.application.experiment import ExperimentContext, expand_experiment
from ux_analyzer.config.loader import load_project
from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.providers.attention_policy import (
    AttentionPolicyConfig,
    ProgressiveAttentionPolicy,
)
from ux_analyzer.providers.prominence import HeuristicProminenceProvider

DEMO_PROJECT = Path(__file__).parents[2] / "benchmarks" / "demo" / "project.yaml"
CI_SEED = 7


def test_legacy_focused_validation_keeps_four_cells_and_zero_model_trial() -> None:
    loaded = load_project(DEMO_PROJECT)
    experiment = next(
        item for item in loaded.project.experiments if item.id == "focused-validation"
    )

    specs = expand_experiment(
        ExperimentContext(
            definition=experiment,
            project=loaded.project,
            config_digest=loaded.config_digest,
        )
    )

    assert len(specs) == 4
    assert {spec.scenario.id for spec in specs} == {"invite-teammate"}
    assert {spec.application_version.id for spec in specs} == {
        "fixture-app-defective",
        "fixture-app-improved",
    }
    assert {spec.policy.value for spec in specs} == {
        "full-list",
        "progressive-prominence-scent",
    }
    assert {spec.prominence_provider_id for spec in specs} == {"heuristic"}
    assert {spec.seed for spec in specs} == {0}
    assert {spec.model_trial for spec in specs} == {0}


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
        return _response(
            {
                "action": action["kind"],
                "element_id": action.get("element_id"),
                "fixture_key": action.get("fixture_key"),
                "direction": action.get("direction"),
                "reason": "Deterministic CI fixture path.",
            }
        )

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


def _reduced_project(path: Path) -> Path:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    scenario = next(item for item in project["scenarios"] if item["id"] == "enable-2fa")
    scenario["budget"].update(
        {
            "max_steps": 60,
            "max_observations": 30,
            "max_interactions": 20,
            "timeout_seconds": 180,
        }
    )
    persona = next(
        item for item in project["personas"] if item["id"] == "first-time-nontechnical"
    )
    persona["attention_temperature"] = 0.4
    project["providers"]["attention"]["coarse_scent_weight"] = 8.0
    experiment = next(
        item for item in project["experiments"] if item["id"] == "core-pair"
    )
    experiment.update(
        {
            "scenario_ids": ["enable-2fa"],
            "application_version_ids": ["fixture-app-improved"],
            "persona_ids": ["first-time-nontechnical"],
            "policies": ["progressive-prominence-scent"],
            "seeds": [CI_SEED],
            "run_count": 1,
        }
    )
    path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    return path


def _prominence_score(events: list[dict[str, object]]) -> dict[str, object]:
    record = next(
        event
        for event in events
        if event.get("kind") == "prominence-recorded"
        and isinstance(event.get("scores"), list)
        and event["scores"]
    )
    scores = record["scores"]
    assert isinstance(scores, list) and scores
    score = scores[0]
    assert isinstance(score, dict)
    return score


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_production_cli_report_is_interactive_and_causal(
    production_servers: tuple[str, str, list[dict[str, object]], dict[str, str]],
    tmp_path: Path,
) -> None:
    fixture_origin, model_base_url, requests, _versions = production_servers
    project = _reduced_project(tmp_path / "project.yaml")
    output = tmp_path / "output"
    result = await asyncio.to_thread(
        CliRunner().invoke,
        cli.app,
        [
            "run",
            str(project),
            "--experiment",
            "core-pair",
            "--output",
            str(output),
            "--fixture-origin",
            fixture_origin,
        ],
        env={
            "UXA_LLM_BASE_URL": model_base_url,
            "UXA_LLM_API_KEY": "ci-api-key",
            "UXA_LLM_TIMEOUT_SECONDS": "30",
            "UXA_SCENT_MODEL": "scent-model",
            "UXA_COGNITIVE_MODEL": "cognitive-model",
            "UXA_REPORT_MODEL": "",
        },
    )
    assert result.exit_code == 0, result.stdout
    assert "run specs: 1" in result.stdout
    assert "finalized runs: 1; execution failures: 0" in result.stdout
    assert "UX samples: 1 valid; 0 invalid" in result.stdout
    assert "report generated:" in result.stdout
    assert requests
    report_path = output / "report.html"
    run_path = next((output / "runs").iterdir())
    persisted = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (run_path / "timeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert persisted["metrics"]["feedback_observed"] is True
    assert "missing-feedback" not in {
        finding["category"] for finding in persisted["findings"]
    }
    score = _prominence_score(events)
    element_id = str(score["element_id"])
    raw_values = score["raw_values"]
    normalized_values = score["normalized_values"]
    contributions = score["feature_contributions"]
    assert isinstance(raw_values, dict)
    assert isinstance(normalized_values, dict)
    assert isinstance(contributions, dict)

    external_requests: list[str] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(service_workers="block")

        async def block_external(route: Route) -> None:
            if route.request.url.startswith(("file:", "data:")):
                await route.continue_()
            else:
                external_requests.append(route.request.url)
                await route.abort()

        await context.route("**/*", block_external)
        page = await context.new_page()
        await page.goto(report_path.resolve().as_uri())
        assert await page.locator("#analysis-summary").is_visible()
        assert await page.locator("#priority-findings").is_visible()
        assert await page.locator("#fix-first").count() == 0
        assert "Recorded signals requiring manual review" in (
            await page.locator("#priority-findings").text_content() or ""
        )
        assert await page.locator("#evidence-workspace").is_visible()
        assert await page.locator("#analysis-summary").evaluate(
            "node => node.compareDocumentPosition(document.querySelector('#comparison-table')) & Node.DOCUMENT_POSITION_FOLLOWING"
        )
        assert "Model review unavailable" in (
            await page.locator("#analysis-summary").text_content() or ""
        )
        assert await page.locator("#priority-findings .finding").count() >= 1
        await page.locator(f'tr[data-run-id="{persisted["run_id"]}"]').click()
        await page.locator('[data-event-kind="prominence-recorded"]').first.click()
        selected = page.locator(f'[data-element-id="{element_id}"]').first
        await selected.hover()
        panel = page.locator("#selected-element-evidence")
        assert await panel.get_attribute("data-selected-element-id") == element_id
        area_row = panel.locator("tr", has_text="Area")
        area_text = await area_row.text_content()
        assert area_text is not None
        assert str(raw_values["area"]) in area_text
        assert str(normalized_values["area"]) in area_text
        assert str(contributions["area"]) in area_text
        await selected.focus()
        assert await panel.get_attribute("data-selected-element-id") == element_id
        await selected.click()
        assert await panel.get_attribute("data-selected-element-id") == element_id
        assert await page.get_by_text("Recorded timeline").count() == 1
        assert await page.get_by_text("Current event").count() == 1
        assert await page.get_by_text("Element evidence").count() == 1
        assert await page.locator("#run-status-banner").count() == 1
        assert await panel.get_by_text("Linked findings").count() == 1
        assert await panel.get_by_text("Linked decisions").count() == 1
        assert await panel.get_by_text("Linked actions and results").count() == 1
        report_payload = await page.locator("#report-data").text_content() or ""
        assert '"selector"' not in report_payload
        assert '"execution_reference"' not in report_payload
        assert "ci-api-key" not in report_payload
        await page.set_viewport_size({"width": 390, "height": 844})
        dimensions = await page.evaluate(
            "({scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth})"
        )
        assert dimensions["scrollWidth"] <= dimensions["innerWidth"]
        assert await page.locator("#analysis-summary").is_visible()
        assert await page.locator("#priority-findings").is_visible()
        await browser.close()

    assert not external_requests


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
