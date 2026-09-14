"""Integration tests for the ``uxa redesign`` command and auto-mode wiring.

Covers plan Task 6: command absent-artifact path, sidecar reuse (no second
browser pass), capture fallback, the ``UXA_REDESIGN_ENABLED`` gate, the
``UXA_REPORT_SYNTHESIS_ENABLED`` env override of the YAML synthesis gate,
and audit page-list unification with the exploration corpus (ADR 0007).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import ux_analyzer.cli as cli
from ux_analyzer.cli import app

DEMO_PROJECT = Path(__file__).parents[3] / "benchmarks" / "demo" / "project.yaml"
runner = CliRunner()


def _result_with_start_url(url: str | None) -> object:
    version = SimpleNamespace(start_url=url)
    spec = SimpleNamespace(application_version=version)
    state = SimpleNamespace(spec=spec)
    return SimpleNamespace(state=state)


def _set_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UXA_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("UXA_LLM_API_KEY", "super-secret-api-key")
    monkeypatch.setenv("UXA_SCENT_MODEL", "scent-model")
    monkeypatch.setenv("UXA_COGNITIVE_MODEL", "cognitive-model")
    monkeypatch.setenv("UXA_LLM_TIMEOUT_SECONDS", "30")


def _capture_payload(url: str) -> dict[str, object]:
    """Minimal page-capture-v2 page (one segment, inventory keys present)."""

    return {
        "schema": "page-capture-v2",
        "url": url,
        "title": "Captured",
        "captured_at": "",
        "viewport": {"width": 1280, "height": 800},
        "document_height": 2000,
        "captured_height": 2000,
        "truncated": False,
        "copy_truncated": False,
        "inventory_truncated": False,
        "segments": [
            {
                "index": 0,
                "y_offset": 0,
                "height": 2000,
                "data_url": "data:image/jpeg;base64,AAAA",
            }
        ],
        "sections": [],
        "headings": [
            {
                "kind": "heading",
                "tag": "h1",
                "label": "Welcome",
                "box": {"x": 0, "y": 0, "w": 640, "h": 48},
                "depth": 0,
            }
        ],
        "forms": [],
        "buttons": [],
        "inputs": [],
        "links": [],
        "paragraphs": [],
    }


def _write_sidecar(output: Path, urls: tuple[str, ...]) -> None:
    pages = [_capture_payload(url) for url in urls]
    (output / "page-capture.json").write_text(
        json.dumps({"schema": "page-capture-v2", "pages": pages}),
        encoding="utf-8",
    )


def _accepted_outcome(urls: tuple[str, ...]) -> Any:
    from ux_analyzer.application.redesign import RedesignPassOutcome
    from ux_analyzer.domain.redesign import RedesignAttempt, RedesignAttemptStatus
    from ux_analyzer.storage.redesign_artifacts import new_attempt_id

    del urls
    return RedesignPassOutcome(
        attempt=RedesignAttempt(
            attempt_id=new_attempt_id(),
            status=RedesignAttemptStatus.ACCEPTED,
            proposals=(),
            killed=(),
            page_understanding=(),
            consistency_notes=(),
            pack_version="redesign-principles-2026-09",
            audience="",
            created_at="2026-09-11T12:00:00Z",
            unavailable_reason="",
            rejection_reasons=(),
        ),
        captures_digest="0" * 64,
    )


def _newest_attempt_payload(output: Path) -> dict[str, object]:
    attempts_root = output / "redesign"
    directories = sorted(entry for entry in attempts_root.iterdir() if entry.is_dir())
    assert len(directories) == 1
    payload = json.loads((directories[0] / "payload.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_redesign_command_reuses_fresh_sidecar_without_second_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_sidecar(output, ("https://example.test/",))
    _set_model_env(monkeypatch)

    async def fake_pass(captures: object, *, audience: str, settings: object) -> Any:
        assert set(captures) == {"https://example.test/"}  # type: ignore[attr-defined]
        assert audience == "designers"
        return _accepted_outcome(tuple(captures))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_run_redesign_pass", fake_pass)

    result = runner.invoke(
        app, ["redesign", str(output), "--audience", "designers"]
    )

    assert result.exit_code == 0, result.output
    payload = _newest_attempt_payload(output)
    assert payload["status"] == "accepted"
    assert "redesign attempt: accepted" in result.output


def test_redesign_command_requires_model_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_sidecar(output, ("https://example.test/",))
    monkeypatch.chdir(tmp_path)
    for name in (
        "UXA_LLM_BASE_URL",
        "UXA_LLM_API_KEY",
        "UXA_SCENT_MODEL",
        "UXA_COGNITIVE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    result = runner.invoke(app, ["redesign", str(output)])

    assert result.exit_code == 1
    assert "model environment" in result.output


def _capture_payload_with_effective_tap_data(url: str) -> dict[str, object]:
    """Like ``_capture_payload`` but with an interactive entry that carries
    its effective tap surface (tap_box), i.e. captured by the current
    pipeline."""

    payload = _capture_payload(url)
    return {
        **payload,
        "buttons": [
            {
                "kind": "button",
                "tag": "button",
                "label": "Play",
                "box": {"x": 0, "y": 0, "w": 80, "h": 30},
                "tap_box": {"x": 0, "y": 0, "w": 320, "h": 240},
            }
        ],
    }


def _write_custom_sidecar(output: Path, pages: list[dict[str, object]]) -> None:
    (output / "page-capture.json").write_text(
        json.dumps({"schema": "page-capture-v2", "pages": pages}),
        encoding="utf-8",
    )


def test_redesign_reuses_sidecar_with_effective_tap_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sidecar whose interactive controls carry tap_box is fresh: no
    second capture, no browser pass."""

    output = tmp_path / "out"
    output.mkdir()
    _write_custom_sidecar(
        output,
        [_capture_payload_with_effective_tap_data("https://example.test/")],
    )
    _set_model_env(monkeypatch)

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("fresh sidecar must not trigger a capture")

    async def fake_pass(captures: object, *, audience: str, settings: object) -> Any:
        del audience, settings
        return _accepted_outcome(tuple(captures))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_capture_page", boom)
    monkeypatch.setattr(cli, "_run_redesign_pass", fake_pass)

    result = runner.invoke(app, ["redesign", str(output)])

    assert result.exit_code == 0, result.output


def test_redesign_recaptures_sidecar_without_effective_tap_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sidecar captured before effective tap-target measurement (a button
    without tap_box) is stale for the redesign consumer: it is re-captured
    so hit-target claims can be validated against the real tappable
    surface."""

    output = tmp_path / "out"
    output.mkdir()
    # A pre-tap-era capture: the interactive control exists but carries no
    # effective tap surface, so hit-target claims cannot be validated
    # against it deterministically.
    legacy = dict(_capture_payload("https://example.test/"))
    legacy["buttons"] = [
        {"kind": "button", "tag": "button", "label": "Play",
         "box": {"x": 0, "y": 0, "w": 80, "h": 30}}
    ]
    _write_custom_sidecar(output, [legacy])
    _set_model_env(monkeypatch)
    captured: list[str] = []

    def fake_capture(url: str, *, max_page_height: int | None = None) -> dict[str, object]:
        del max_page_height
        captured.append(url)
        return _capture_payload_with_effective_tap_data(url)

    async def fake_pass(captures: object, *, audience: str, settings: object) -> Any:
        del audience, settings
        return _accepted_outcome(tuple(captures))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_capture_page", fake_capture)
    monkeypatch.setattr(cli, "_run_redesign_pass", fake_pass)

    result = runner.invoke(app, ["redesign", str(output)])

    assert result.exit_code == 0, result.output
    assert captured == ["https://example.test/"]
    # The refreshed sidecar now carries the effective tap surfaces.
    sidecar = json.loads((output / "page-capture.json").read_text(encoding="utf-8"))
    button = sidecar["pages"][0]["buttons"][0]
    assert "tap_box" in button


def test_redesign_command_reports_absent_captures(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()

    result = runner.invoke(app, ["redesign", str(output)])

    assert result.exit_code == 1
    assert "no page captures" in result.output


def test_redesign_command_captures_when_sidecar_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _set_model_env(monkeypatch)
    captured: list[str] = []

    def fake_capture(url: str, *, max_page_height: int | None = None) -> dict[str, object]:
        del max_page_height
        captured.append(url)
        return _capture_payload(url)

    async def fake_pass(captures: object, *, audience: str, settings: object) -> Any:
        del audience, settings
        return _accepted_outcome(tuple(captures))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_capture_page", fake_capture)
    monkeypatch.setattr(cli, "_run_redesign_pass", fake_pass)

    result = runner.invoke(
        app,
        [
            "redesign",
            str(output),
            "--pages",
            "https://example.test/",
            "https://example.test/pricing",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == ["https://example.test/", "https://example.test/pricing"]
    sidecar = json.loads((output / "page-capture.json").read_text(encoding="utf-8"))
    assert [page["url"] for page in sidecar["pages"]] == [
        "https://example.test/",
        "https://example.test/pricing",
    ]


def test_unavailable_attempt_persists_model_call_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal model failure publishes an unavailable attempt *plus* the
    sanitized transport records, so the failure stays debuggable offline."""

    from dataclasses import replace

    from ux_analyzer.adapters.openai import (
        ModelCallRecord,
        ModelRole,
        RetryEvent,
        TokenUsage,
    )
    from ux_analyzer.domain.redesign import RedesignAttemptStatus

    output = tmp_path / "out"
    output.mkdir()
    _write_sidecar(output, ("https://example.test/",))
    _set_model_env(monkeypatch)

    async def failing_pass(
        captures: object, *, audience: str, settings: object
    ) -> Any:
        del audience, settings
        outcome = _accepted_outcome(tuple(captures))  # type: ignore[arg-type]
        unavailable = replace(
            outcome,
            attempt=replace(
                outcome.attempt,
                status=RedesignAttemptStatus.UNAVAILABLE,
                unavailable_reason=(
                    "proposer failed for https://example.test/: "
                    "ModelFailureError: invalid structured output"
                ),
            ),
            model_call_records=(
                ModelCallRecord(
                    role=ModelRole.REDESIGN_PROPOSER,
                    model="deepseek-v4.1-flash",
                    endpoint_origin="https://llm.example.test",
                    prompt_digest="0" * 64,
                    schema_version="redesign-proposer-v1",
                    attempts=3,
                    latency_ms=1200,
                    token_usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    request={},
                    response={
                        "failure": "invalid structured output",
                        "provider": {"status_code": 200},
                        "diagnostics": {
                            "stage": "schema_validation",
                            "response_mode": "json-object",
                            "attempt_count": 3,
                        },
                    },
                    retries=(
                        RetryEvent(
                            role=ModelRole.REDESIGN_PROPOSER,
                            model="deepseek-v4.1-flash",
                            attempt=1,
                            reason="invalid-structured-output",
                            status_code=200,
                            delay_seconds=0.25,
                        ),
                    ),
                ),
            ),
        )
        return unavailable

    monkeypatch.setattr(cli, "_run_redesign_pass", failing_pass)

    result = runner.invoke(app, ["redesign", str(output)])

    assert result.exit_code == 0, result.output
    payload = _newest_attempt_payload(output)
    assert payload["status"] == "unavailable"
    records_path = next((output / "redesign").iterdir()) / "model-calls.json"
    assert records_path.is_file()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert records["schema"] == "redesign-model-calls-v1"
    call = records["model_calls"][0]
    assert call["role"] == "redesign-proposer"
    assert call["attempts"] == 3
    assert call["retries"][0]["reason"] == "invalid-structured-output"
    assert call["failure"]["reason"] == "invalid structured output"
    assert call["failure"]["diagnostics"]["stage"] == "schema_validation"
    assert "request" not in call and "response" not in call


def test_redesign_dry_run_prints_page_list_without_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_sidecar(
        output,
        ("https://example.test/", "https://example.test/pricing"),
    )

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("models must not run in dry-run")

    monkeypatch.setattr(cli, "_run_redesign_pass", boom)

    result = runner.invoke(app, ["redesign", str(output), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "https://example.test/" in result.stdout
    assert "https://example.test/pricing" in result.stdout


def test_redesign_respects_max_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    _write_sidecar(
        output,
        (
            "https://example.test/",
            "https://example.test/pricing",
            "https://example.test/about",
        ),
    )

    result = runner.invoke(
        app, ["redesign", str(output), "--dry-run", "--max-pages", "2"]
    )

    assert result.exit_code == 0, result.output
    assert "https://example.test/pricing" in result.stdout
    assert "https://example.test/about" not in result.stdout


def test_auto_redesign_gated_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = (_result_with_start_url("https://example.test/"),)
    captured: list[str] = []

    def fake_capture(url: str, *, max_page_height: int | None = None) -> dict[str, object]:
        del max_page_height
        captured.append(url)
        return _capture_payload(url)

    async def fake_pass(captures: object, *, audience: str, settings: object) -> Any:
        del audience, settings
        return _accepted_outcome(tuple(captures))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_capture_page", fake_capture)
    monkeypatch.setattr(cli, "_run_redesign_pass", fake_pass)
    _set_model_env(monkeypatch)

    # Gate off (default): no redesign attempt is produced.
    monkeypatch.delenv("UXA_REDESIGN_ENABLED", raising=False)
    off_output = tmp_path / "off"
    off_output.mkdir()
    cli._run_auto_redesign(output=off_output, results=results)
    assert not (off_output / "redesign").exists()
    assert captured == []

    # Gate on: one capture pass over the shared page list, then the attempt.
    monkeypatch.setenv("UXA_REDESIGN_ENABLED", "1")
    on_output = tmp_path / "on"
    on_output.mkdir()
    cli._run_auto_redesign(output=on_output, results=results)
    assert captured == ["https://example.test/"]
    payload = _newest_attempt_payload(on_output)
    assert payload["status"] == "accepted"


def test_auto_redesign_wired_into_completion_paths() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    wiring = [
        line.strip()
        for line in source.splitlines()
        if "_run_auto_redesign(" in line and "def _run_auto_redesign" not in line
    ]
    assert len(wiring) >= 2, "auto redesign must run after audit+pagespeed"


def test_synthesis_env_override_of_yaml_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": False}
    disabled_path = tmp_path / "disabled.yaml"
    disabled_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    project["evaluation"]["report_synthesis"] = {"enabled": True}
    enabled_path = tmp_path / "enabled.yaml"
    enabled_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.delenv("UXA_REPORT_SYNTHESIS_ENABLED", raising=False)

    # Unset: current behavior (YAML gate decides).
    assert cli._synthesis_gate(cli._load_project_or_exit(disabled_path)) is False
    assert cli._synthesis_gate(cli._load_project_or_exit(enabled_path)) is True

    # Set: the environment overrides YAML in both directions.
    monkeypatch.setenv("UXA_REPORT_SYNTHESIS_ENABLED", "1")
    assert cli._synthesis_gate(cli._load_project_or_exit(disabled_path)) is True
    monkeypatch.setenv("UXA_REPORT_SYNTHESIS_ENABLED", "0")
    assert cli._synthesis_gate(cli._load_project_or_exit(enabled_path)) is False


class _FakeExplorationStore:
    """Doubles ExplorationArtifactStore with one finalized attempt."""

    def __init__(self, output: Path, *, finalized: bool = True) -> None:
        del output
        self._finalized = finalized

    def get_index(self) -> dict[str, object]:
        if not self._finalized:
            return {"attempts": []}
        return {
            "attempts": [
                {
                    "attempt_id": "2026-09-11T120000Z-abcdef123456-1",
                    "status": "succeeded",
                }
            ]
        }

    def load_attempt(self, attempt_id: str) -> dict[str, object]:
        assert attempt_id == "2026-09-11T120000Z-abcdef123456-1"
        return {
            "corpus": {
                "pages": [
                    {"url": "https://example.test/"},
                    {"url": "https://example.test/pricing"},
                ],
                "link_graph": {
                    "https://example.test/": ["https://example.test/pricing"]
                },
            }
        }


def test_audit_page_list_uses_exploration_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audited: list[list[str]] = []

    def fake_sync(
        urls: object, *, capture_hook: object | None = None
    ) -> dict[str, Any]:
        del capture_hook
        audited.append(list(urls))  # type: ignore[arg-type]
        return {
            "schema_version": "ux-audit-v1",
            "total_issues": 0,
            "urls": [],
            "errors": [],
        }

    monkeypatch.setattr(cli, "ExplorationArtifactStore", _FakeExplorationStore)
    monkeypatch.setattr(cli, "_ux_audit_sync", fake_sync)

    # With a finalized exploration attempt: the shared page list includes
    # BFS-discovered corpus pages beyond the start URLs (ADR 0007).
    cli._write_ux_audit(tmp_path, (_result_with_start_url("https://example.test/"),))
    assert audited == [["https://example.test/", "https://example.test/pricing"]]

    # Without a finalized attempt: exactly the start URLs (current behavior).
    monkeypatch.setattr(
        cli, "ExplorationArtifactStore", lambda output: _FakeExplorationStore(output, finalized=False)
    )
    audited.clear()
    cli._write_ux_audit(tmp_path, (_result_with_start_url("https://example.test/"),))
    assert audited == [["https://example.test/"]]


def test_synthesis_gate_treats_empty_env_value_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dotenv turns ``UXA_REPORT_SYNTHESIS_ENABLED=`` into an empty string;
    an empty value must behave exactly like unset, never silently flipping
    the project YAML gate (finding: .env.example shipped an empty value)."""

    project = yaml.safe_load(DEMO_PROJECT.read_text(encoding="utf-8"))
    assert isinstance(project, dict)
    project.setdefault("evaluation", {})["report_synthesis"] = {"enabled": True}
    enabled_path = tmp_path / "enabled.yaml"
    enabled_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    project["evaluation"]["report_synthesis"] = {"enabled": False}
    disabled_path = tmp_path / "disabled.yaml"
    disabled_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    monkeypatch.setenv("UXA_REPORT_SYNTHESIS_ENABLED", "")
    assert cli._synthesis_gate(cli._load_project_or_exit(enabled_path)) is True
    assert cli._synthesis_gate(cli._load_project_or_exit(disabled_path)) is False


@pytest.mark.parametrize(
    "env_value,expected_height",
    [("", 12000), ("0", 12000), ("bogus", 12000), ("800", 800)],
)
def test_redesign_max_page_height_env_never_crashes_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected_height: int,
) -> None:
    """The same env input behaves the same everywhere: unset, unparseable,
    or non-positive ``UXA_REDESIGN_MAX_PAGE_HEIGHT`` falls back to the
    default instead of crashing the standalone capture path."""

    output = tmp_path / "out"
    _set_model_env(monkeypatch)
    seen: list[int | None] = []

    def fake_capture(url: str, *, max_page_height: int | None = None) -> dict[str, object]:
        seen.append(max_page_height)
        return _capture_payload(url)

    async def fake_pass(captures: object, *, audience: str, settings: object) -> Any:
        del audience, settings
        return _accepted_outcome(tuple(captures))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_capture_page", fake_capture)
    monkeypatch.setattr(cli, "_run_redesign_pass", fake_pass)
    monkeypatch.setenv("UXA_REDESIGN_MAX_PAGE_HEIGHT", env_value)

    result = runner.invoke(
        app,
        ["redesign", str(output), "--pages", "https://example.test/"],
    )
    assert result.exit_code == 0, result.output
    assert seen == [expected_height]


def test_redesign_pages_are_normalized_deduped_and_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit --pages take the canonical form (fragment stripped, host
    lowercased) so they hit the persisted sidecar instead of re-capturing a
    near-duplicate key; the cap applies to the override too."""

    output = tmp_path / "out"
    _set_model_env(monkeypatch)
    captured: list[str] = []

    def fake_capture(url: str, *, max_page_height: int | None = None) -> dict[str, object]:
        del max_page_height
        captured.append(url)
        return _capture_payload(url)

    async def fake_pass(captures: object, *, audience: str, settings: object) -> Any:
        del audience, settings
        return _accepted_outcome(tuple(captures))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_capture_page", fake_capture)
    monkeypatch.setattr(cli, "_run_redesign_pass", fake_pass)

    result = runner.invoke(
        app,
        [
            "redesign",
            str(output),
            "--pages",
            "https://Example.test/#top",
            "https://example.test/pricing",
            "https://example.test/",
            "--max-pages",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == ["https://example.test/", "https://example.test/pricing"]


def test_redesign_rejects_invalid_pages_url(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()

    result = runner.invoke(
        app, ["redesign", str(output), "--pages", "not-a-url"]
    )

    assert result.exit_code == 1
    assert "invalid page URL" in result.output
    assert "not-a-url" in result.output
