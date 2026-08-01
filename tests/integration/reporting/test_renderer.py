from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from playwright.async_api import Route, async_playwright

import ux_analyzer.reporting.renderer as renderer
from ux_analyzer.domain.run import RunStarted
from ux_analyzer.ports.artifacts import BundleManifest, RedactionPolicy
from ux_analyzer.reporting.renderer import render_experiment_report
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_checksums(run: Path) -> None:
    files = sorted(
        path
        for path in run.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (run / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(run).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _write_run(
    root: Path,
    run_id: str,
    *,
    version: str,
    discovery_cost: float,
    screenshot: bytes = b"not-an-image",
    outcome: str = "verified-success",
    verified: bool | None = None,
    terminal_reason: str | None = None,
    evaluation_failure_reason: str | None = None,
    ux_sample_valid: bool | None = None,
    ux_sample_invalid_reason: str | None = None,
) -> None:
    is_verified = outcome == "verified-success" if verified is None else verified
    is_valid_sample = (
        outcome in {"verified-success", "agent-abandoned", "budget-exhausted"}
        and evaluation_failure_reason is None
        if ux_sample_valid is None
        else ux_sample_valid
    )
    run = root / "runs" / run_id
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "screenshot.png").write_bytes(screenshot)
    _write_json(
        run / "manifest.json",
        {
            "run_id": run_id,
            "seed": 7,
            "config_digest": "config-sha",
            "endpoint_origin": "https://llm.example.test/v1",
            "model_ids": {"cognitive": "model-v1"},
            "prompt_versions": {"cognitive": "cognitive-v1"},
            "package_version": "0.1.0",
            "provider_versions": {"observation": "fixture-v1"},
            "provider_manifests": [
                {
                    "provider_id": "provider",
                    "role": "cognitive",
                    "model_id": "model-v1",
                    "endpoint_origin": "https://llm.example.test",
                    "version": "1",
                }
            ],
        },
    )
    events = [
        {
            "sequence": 1,
            "kind": "viewport-captured",
            "viewport_width": 800,
            "viewport_height": 600,
            "snapshot": {
                "id": "viewport-1",
                "screenshot_artifact": "artifacts/screenshot.png",
                "elements": [
                    {
                        "id": "target",
                        "role": "button",
                        "label": '<img src=x onerror="alert(1)">',
                        "bounds": {"x": 40, "y": 50, "width": 180, "height": 40},
                        "visibility_fraction": 1,
                        "occlusion_fraction": 0.25,
                        "local_contrast": 0.75,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                        "selector": "button[data-testid=secret]",
                        "test_id": "secret",
                        "execution_reference": {
                            "provider_id": "provider",
                            "token": "secret-token",
                        },
                    },
                    {
                        "id": "competitor",
                        "role": "button",
                        "label": "Share",
                        "bounds": {"x": 300, "y": 50, "width": 100, "height": 40},
                        "visibility_fraction": 1,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                    },
                ],
                "regions": [{"id": "team", "label": "Team"}],
            },
        },
        {
            "sequence": 2,
            "kind": "observation-recorded",
            "observation": {
                "viewport_id": "viewport-1",
                "newly_revealed_elements": [
                    {
                        "id": "target",
                        "role": "button",
                        "label": '<img src=x onerror="alert(1)">',
                        "bounds": {"x": 40, "y": 50, "width": 180, "height": 40},
                        "visibility_fraction": 1,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                    }
                ],
                "remembered_elements": [],
                "region_context": {"id": "team", "label": "Team"},
            },
        },
        {
            "sequence": 3,
            "kind": "prominence-recorded",
            "viewport_id": "viewport-1",
            "scores": [
                {
                    "element_id": "target",
                    "raw_score": 0.2,
                    "normalized_probability": 0.3,
                    "feature_contributions": {"area": 0.1, "contrast": 0.2},
                    "raw_values": {"area": 7200, "contrast": 4.5},
                    "normalized_values": {"area": 0.4, "contrast": 0.8},
                }
            ],
        },
        {
            "sequence": 4,
            "kind": "coarse-scent-recorded",
            "scores": [{"element_id": "target", "score": 0.4}],
        },
        {
            "sequence": 5,
            "kind": "full-scent-recorded",
            "scores": [{"element_id": "target", "score": 0.6}],
        },
        {
            "sequence": 6,
            "kind": "action-proposed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "reason": "Target matches goal.",
        },
        {
            "sequence": 7,
            "kind": "action-executed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "succeeded": True,
            "viewport_id": "viewport-1",
            "execution_reference": {"token": "secret-token"},
        },
        {
            "sequence": 8,
            "kind": "verification-recorded",
            "result": {
                "verified": is_verified,
                "evidence_ids": ["verify-1"],
                "details": (
                    "Independent verifier passed."
                    if is_verified
                    else "Independent verifier did not confirm completion."
                ),
            },
        },
        {
            "sequence": 9,
            "kind": "model-call-recorded",
            "record": {
                "role": "cognitive",
                "model": "model-v1",
                "endpoint_origin": "https://llm.example.test",
                "prompt_digest": "prompt-sha",
                "schema_version": "cognitive-v1",
                "attempts": 2,
                "latency_ms": 125,
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
                "request": {"messages": [{"role": "user", "content": "safe request"}]},
                "response": {"summary": "safe response"},
                "retries": [{"attempt": 1, "reason": "rate-limit"}],
            },
        },
        {
            "sequence": 10,
            "kind": "run-terminated",
            "outcome": {"kind": outcome},
        },
    ]
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_json(
        run / "result.json",
        {
            "run_id": run_id,
            "agent_claimed_success": True,
            "outcome": {"kind": outcome},
            "terminal_reason": terminal_reason,
            "evaluation_failure_reason": evaluation_failure_reason,
            "ux_sample_valid": is_valid_sample,
            "ux_sample_invalid_reason": ux_sample_invalid_reason,
            "evidence": {
                "prominence": [],
                "scent": [],
                "selections": [],
                "decisions": [],
                "model_calls": [],
                "screenshot_artifacts": [],
            },
            "metrics": {
                "scenario_id": "invite",
                "application_version_id": version,
                "persona_id": "persona",
                "policy": "progressive-prominence-scent",
                "verified_completion": is_verified,
                "wrong_actions": 0 if version == "improved" else 2,
                "backtracks": 0,
                "discovery_cost": {"total": discovery_cost},
                "evidence": [
                    {
                        "evidence_id": f"{run_id}:discovery-cost",
                        "evidence_class": "model-estimate",
                        "description": "Seeded discovery cost.",
                    },
                    {
                        "evidence_id": f"{run_id}:human",
                        "evidence_class": "unsupported-human-claim",
                        "description": "People will love it.",
                    },
                ],
            },
            "findings": [
                {
                    "finding_id": f"{run_id}:weak-scent",
                    "category": "weak-scent",
                    "title": "Target wording gives weak goal cues",
                    "cause": "Target scent 0.2 is below configured threshold 0.3.",
                    "severity": "medium",
                    "reproducibility": "model-dependent",
                    "evidence_class": "model-estimate",
                    "evidence_ids": [f"{run_id}:discovery-cost"],
                    "limitations": ["simulated benchmark evidence"],
                    "run_ids": [run_id],
                    "viewport_ids": ["viewport-1"],
                    "element_ids": ["target"],
                    "supporting_metrics": {"target-scent": 0.2},
                    "action_sequence": ["interact-with-element target: succeeded"],
                    "replay_links": [f"#run={run_id}&element=target"],
                }
            ],
            "limitations": ["Simulated benchmark; not human satisfaction evidence."],
        },
    )
    _write_checksums(run)


def test_renderer_embeds_sanitized_replay_evidence_and_controls(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "run-1",
        version="defective",
        discovery_cost=8,
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")

    html = output.read_text(encoding="utf-8")
    assert output == tmp_path / "report.html"
    assert "Comparison overview" in html
    assert "User actions" in html
    assert "Observations" in html
    assert "Discovery cost" in html
    assert "deterministic-fact" in html
    assert "model-estimate" in html
    assert "unsupported-human-claim" in html
    assert "records measured interface, action, geometry, bundle" in html
    assert "configured heuristic, scent, policy, persona, memory" in html
    assert "Run workspace" in html
    assert "Recorded timeline" in html
    assert "play-pause" in html
    assert "restart-playback" in html
    assert "Prominence contributions" in html
    assert "Element evidence" in html
    assert "Observation" in html
    assert "Terminal / failure" in html
    assert "Scent" in html
    assert "Model decision and reason" in html
    assert "Action and result" in html
    assert "Verification" in html
    assert "Memory" in html
    assert "Prompt version" in html
    assert "Schema version" in html
    assert "Sanitized request summary" in html
    assert '"bundle_path":"runs\\u002frun-1"' in html
    assert '"integrity_status":"trusted"' in html
    assert "Trust boundaries and limitations" in html
    assert "Seeded discovery cost." in html
    assert '"width":800' in html
    assert "secret-token" not in html
    assert "data-testid=secret" not in html
    assert "<img src=x onerror" not in html
    assert "fetch(" not in html
    assert '<link rel="stylesheet"' not in html
    assert "<script src=" not in html


def test_overview_counts_executed_actions_and_formats_discovery_cost(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-effort",
        version="improved",
        discovery_cost=4.199999999999999,
    )
    run = tmp_path / "runs" / "run-effort"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events.insert(
        -1,
        {
            "sequence": 10,
            "kind": "decision-recorded",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "reason": "Duplicate representation of the proposed action.",
        },
    )
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    _write_checksums(run)

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert 'data-metric="user-actions">1</td>' in html
    assert 'data-metric="observations">1</td>' in html
    assert 'data-metric="discovery-cost">4.2</td>' in html
    assert 'data-metric="estimated-task-seconds">2.5 s</td>' in html
    assert 'data-metric="model-calls">1</td>' in html
    assert 'data-metric="analysis-latency">125 ms</td>' in html
    assert 'data-metric="analysis-tokens">19</td>' in html
    assert "simulated-task-time-v1" in html
    assert "Analysis cost is not user effort" in html
    assert "Monetary estimate unavailable" in html


def test_renderer_exposes_safe_model_and_progress_diagnostics(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "run-diagnostics",
        version="improved",
        discovery_cost=3,
        outcome="model-failure",
        verified=False,
        terminal_reason="cognitive: inspect action requires element_id",
        ux_sample_valid=False,
        ux_sample_invalid_reason="model-failure: invalid cognitive response",
    )
    run = tmp_path / "runs" / "run-diagnostics"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    terminal = events.pop()
    events.extend(
        [
            {
                "kind": "attention-selection-recorded",
                "viewport_id": "viewport-2",
                "selected_ids": ["email", "submit"],
                "recovery_selected_ids": ["submit"],
                "selection_mode": "recovery-forced",
                "region_id": None,
                "element_probabilities": {"email": 0.4, "submit": 0.1},
                "region_probabilities": {"form": 0.4, "actions": 0.1},
            },
            {
                "kind": "model-failure",
                "role": "cognitive",
                "reason": "inspect action requires element_id",
                "response_summary": {
                    "action": "inspect",
                    "element_id": None,
                    "api_key": "provider-secret",
                },
            },
            {
                "kind": "repeated-fixture-input",
                "element_id": "target",
                "fixture_key": "invite_email",
                "reason": "fixture input was already entered successfully",
            },
            {
                "kind": "repeated-action-detected",
                "action": {"kind": "interact-with-element", "element_id": "target"},
                "count": 3,
                "reason": "identical action repeated without sufficient progress",
            },
            {
                "kind": "no-progress-recovery",
                "action": {"kind": "scroll", "direction": "down"},
                "count": 2,
                "reason": "action succeeded but the recaptured interface did not change semantically",
            },
            {
                "kind": "no-progress-detected",
                "count": 3,
                "reason": "three consecutive actions produced no state progress",
            },
            {
                "kind": "repeated-action-cycle",
                "cycle_length": 2,
                "reason": "repeated semantic action cycle detected",
                "url": "https://fixture.test/invite?token=private-secret",
            },
        ]
    )
    events.append(terminal)
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    _write_checksums(run)

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert "inspect action requires element_id" in html
    assert '"ux_sample_valid":false' in html
    assert "model-failure: invalid cognitive response" in html
    assert '"recovery_selected_ids":["submit"]' in html
    assert "recovery-forced" in html
    assert '"role":"cognitive"' in html
    assert '"response_summary":{"action":"inspect","element_id":null}' in html
    assert "repeated-fixture-input" in html
    assert "invite_email" in html
    assert "repeated-action-detected" in html
    assert '"count":3' in html
    assert "no-progress-recovery" in html
    assert (
        '"kind":"no-progress-recovery",'
        '"action":{"kind":"scroll","direction":"down"},'
        '"count":2'
    ) in html
    assert "no-progress-detected" in html
    assert "repeated-action-cycle" in html
    assert '"cycle_length":2' in html
    assert "provider-secret" not in html
    assert "private-secret" not in html


def test_renderer_includes_all_failed_experiment_and_staging_crash(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".staging" / "run-crashed"
    staging.mkdir(parents=True)
    _write_json(staging / "manifest.json", {"run_id": "run-crashed", "seed": 7})
    (staging / "timeline.jsonl").write_text(
        json.dumps({"sequence": 1, "kind": "run-started", "run_id": "run-crashed"})
        + "\n"
        + '{"sequence":2,"kind":"viewport-captured"',
        encoding="utf-8",
    )
    _write_json(
        staging / "crash.marker",
        {"run_id": "run-crashed", "reason": "browser capture failed"},
    )
    _write_json(
        staging / "result.json",
        {
            "metrics": {
                "discovery_cost": {"total": 999999},
                "verified_completion": True,
            },
            "outcome": {"kind": "verified-success"},
        },
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "run_metrics": [],
            "cell_aggregates": [],
            "variant_comparisons": [],
            "findings": {},
            "failures": [
                {
                    "run_id": "run-crashed",
                    "error_type": "ProviderFailure",
                    "stage": "execution",
                    "terminal_state": "crashed",
                    "reason": "browser capture failed",
                    "scenario_id": "enable-2fa",
                    "application_version_id": "fixture-app-improved",
                    "persona_id": "impatient",
                    "policy": "progressive-prominence-scent",
                    "seed": 7,
                }
            ],
        },
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert output.is_file()
    assert "Comparison overview" in html
    assert 'data-run-id="run-crashed"' in html
    assert "browser capture failed" in html
    assert "run-crashed" in html
    assert "enable-2fa" in html
    assert "progressive-prominence-scent" in html
    assert "999999" not in html


def test_renderer_does_not_score_incomplete_bundle_with_result_metrics(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".staging" / "run-incomplete"
    staging.mkdir(parents=True)
    _write_json(staging / "manifest.json", {"run_id": "run-incomplete", "seed": 8})
    (staging / "timeline.jsonl").write_text(
        json.dumps({"sequence": 1, "kind": "run-started"}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        staging / "result.json",
        {"metrics": {"discovery_cost": {"total": 888888}}},
    )
    _write_json(staging / "crash.marker", {"reason": "capture failed"})

    output = render_experiment_report(tmp_path, tmp_path / "report.html")

    assert "888888" not in output.read_text(encoding="utf-8")


def test_renderer_excludes_tampered_bundle_from_scorecards_and_gates(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-defective", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run-improved", version="improved", discovery_cost=3)
    tampered = tmp_path / "runs" / "run-improved" / "result.json"
    tampered.write_text(
        tampered.read_text(encoding="utf-8").replace('"total": 3', '"total": 999999'),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "variant_comparisons": [
                {
                    "baseline": {
                        "scenario_id": "invite",
                        "application_version_id": "defective",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "improved": {
                        "scenario_id": "invite",
                        "application_version_id": "improved",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "gate": {
                        "passed": True,
                        "paired_seed_count": 1,
                        "reasons": [],
                    },
                }
            ]
        },
    )

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert "checksum mismatch: result.json" in html
    assert "999999" not in html
    assert "All directional checks passed" not in html


def test_renderer_derives_directional_gate_from_all_trusted_bundles(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-defective", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run-improved", version="improved", discovery_cost=3)

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "All directional checks passed" in html
    assert "Directional gates unavailable" not in html


def test_renderer_excludes_active_bundle_from_scorecards_and_findings(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-active", version="improved", discovery_cost=777777)
    active_run = tmp_path / "runs" / "run-active"
    (active_run / ".active").write_text('{"run_id":"run-active"}', encoding="utf-8")

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert "run-active" in html
    assert "active bundle marker present" in html
    assert "777777" not in html
    assert "Target wording gives weak goal cues" not in html


@pytest.mark.parametrize("filename", ("manifest.json", "timeline.jsonl", "result.json"))
def test_renderer_reports_missing_required_bundle_file_as_untrusted(
    tmp_path: Path, filename: str
) -> None:
    _write_run(tmp_path, "run-missing", version="defective", discovery_cost=8)
    (tmp_path / "runs" / "run-missing" / filename).unlink()

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert f"missing required bundle file: {filename}" in html
    assert "run-missing" in html


def test_renderer_never_embeds_sensitive_fixture_artifacts(tmp_path: Path) -> None:
    invite_email = "invitee@example.test"
    totp_code = "246810"
    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        BundleManifest(
            run_id="run-sensitive",
            seed=1,
            config_digest="config-sha",
            endpoint_origin="https://llm.example.test",
        ),
        redaction=RedactionPolicy(exact_values=(invite_email, totp_code)),
    )
    screenshot = writer.write_artifact(
        "screenshot.png",
        b"\x89PNG\r\n\x1a\n" + invite_email.encode() + totp_code.encode(),
    )
    writer.append_event(RunStarted(run_id="run-sensitive"))
    writer.append_event(
        {
            "kind": "viewport-captured",
            "snapshot": {
                "id": "viewport-1",
                "screenshot_artifact": screenshot.path,
                "elements": [],
            },
        }
    )
    writer.append_event(
        {
            "kind": "run-terminated",
            "outcome": {"kind": "agent-abandoned"},
            "fixture_inputs": {
                "invite_email": invite_email,
                "totp_code": totp_code,
            },
        }
    )
    final_path = writer.finalize(
        {
            "outcome": {"kind": "agent-abandoned"},
            "invite_email": invite_email,
            "totp_code": totp_code,
        }
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert invite_email not in html
    assert totp_code not in html
    assert base64.b64encode(invite_email.encode()).decode() not in html
    assert base64.b64encode(totp_code.encode()).decode() not in html
    artifact_bytes = b"".join(
        path.read_bytes() for path in (final_path / "artifacts").iterdir()
    )
    assert invite_email.encode() not in artifact_bytes
    assert totp_code.encode() not in artifact_bytes


def test_renderer_builds_comparison_and_splits_large_experiment(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-defective", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run-improved", version="improved", discovery_cost=3)
    _write_json(
        tmp_path / "experiment.json",
        {
            "variant_comparisons": [
                {
                    "baseline": {
                        "scenario_id": "invite",
                        "application_version_id": "defective",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "improved": {
                        "scenario_id": "invite",
                        "application_version_id": "improved",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "gate": {
                        "passed": True,
                        "paired_seed_count": 1,
                        "reasons": [],
                    },
                }
            ]
        },
    )

    output = render_experiment_report(
        tmp_path,
        tmp_path / "experiment.html",
        max_single_file_bytes=100,
    )

    html = output.read_text(encoding="utf-8")
    run_pages = tmp_path / "experiment-runs"
    assert output == tmp_path / "experiment.html"
    assert "run-defective.html" in html
    assert "run-improved.html" in html
    assert (run_pages / "run-defective.html").is_file()
    assert (run_pages / "run-improved.html").is_file()
    assert "Defective" in html
    assert "Improved" in html
    assert "discovery-cost" in html
    assert "directional gates" in html
    assert "All directional checks passed" in html


def test_renderer_preflights_threshold_before_full_aggregate_render(
    tmp_path: Path, monkeypatch
) -> None:
    _write_run(
        tmp_path,
        "run-large",
        version="defective",
        discovery_cost=8,
        screenshot=b"x" * 50_000,
    )
    aggregate_payloads: list[bool] = []
    original_render = renderer._render_html

    def track_render(context, title):
        aggregate_payloads.append(any("timeline" in run for run in context["runs"]))
        return original_render(context, title)

    monkeypatch.setattr(renderer, "_render_html", track_render)

    render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        max_single_file_bytes=20_000,
    )

    assert aggregate_payloads[0] is False


def test_renderer_replaces_oversized_run_page_with_bounded_notice(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-oversized",
        version="defective",
        discovery_cost=8,
        screenshot=b"x" * 100_000,
    )
    threshold = 20_000

    render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        max_single_file_bytes=threshold,
    )

    run_page = tmp_path / "report-runs" / "run-oversized.html"
    html = run_page.read_text(encoding="utf-8")
    assert run_page.stat().st_size <= threshold
    assert (
        "Detailed replay omitted because run page exceeds configured size limit."
        in html
    )


def test_renderer_streams_checksum_verification_for_unreferenced_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(tmp_path, "run-streamed", version="defective", discovery_cost=8)
    artifact = tmp_path / "runs" / "run-streamed" / "artifacts" / "large.bin"
    artifact.write_bytes(b"x" * 100_000)
    _write_checksums(artifact.parents[1])
    original_read_bytes = Path.read_bytes

    def reject_whole_file_read(path: Path) -> bytes:
        if path == artifact:
            raise AssertionError("checksum verification read whole artifact")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)

    output = render_experiment_report(tmp_path, tmp_path / "report.html")

    assert output.is_file()


def test_renderer_split_index_is_concise_and_uses_collision_safe_run_links(
    tmp_path: Path,
) -> None:
    for run_id in ("run active", "run_active"):
        _write_run(tmp_path, run_id, version="defective", discovery_cost=8)
        result_path = tmp_path / "runs" / run_id / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["limitations"] = ["x" * 120_000]
        _write_json(result_path, result)
        _write_checksums(result_path.parent)
    threshold = 80_000

    output = render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        max_single_file_bytes=threshold,
    )

    html = output.read_text(encoding="utf-8")
    run_pages = tuple((tmp_path / "report-runs").glob("*.html"))
    assert output.stat().st_size <= threshold
    assert "x" * 1_000 not in html
    assert len(run_pages) == 2
    assert len({path.name for path in run_pages}) == 2
    assert all(path.name in html for path in run_pages)


def test_renderer_overview_keeps_every_run_and_preserves_outcome_from_failure_stage(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-timeout",
        version="defective",
        discovery_cost=12,
        outcome="timed-out",
        verified=False,
        terminal_reason="run timeout exceeded",
    )
    _write_run(
        tmp_path,
        "run-evaluation",
        version="improved",
        discovery_cost=4,
        outcome="agent-abandoned",
        verified=False,
        terminal_reason="agent chose to stop",
        evaluation_failure_reason="evaluation evidence unavailable: target absent",
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "failures": [
                {
                    "run_id": "run-evaluation",
                    "error_type": "EvaluationFailure",
                    "stage": "evaluation",
                    "terminal_state": "finalized",
                    "reason": "evaluation evidence unavailable: target absent",
                    "scenario_id": "invite",
                    "application_version_id": "improved",
                    "persona_id": "persona",
                    "policy": "progressive-prominence-scent",
                    "seed": 7,
                }
            ]
        },
    )

    experiment = renderer._load_experiment(tmp_path)
    by_id = {run["run_id"]: run for run in experiment["runs"]}
    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert len(experiment["runs"]) == 2
    assert by_id["run-timeout"]["outcome"] == "timed-out"
    assert by_id["run-timeout"]["terminal_reason"] == "run timeout exceeded"
    assert by_id["run-evaluation"]["outcome"] == "agent-abandoned"
    assert by_id["run-evaluation"]["stage"] == "evaluation"
    assert by_id["run-evaluation"]["evaluation_failure_reason"] == (
        "evaluation evidence unavailable: target absent"
    )
    assert 'data-run-id="run-timeout"' in html
    assert 'data-run-id="run-evaluation"' in html
    assert "timed-out" in html
    assert "agent-abandoned" in html
    assert "evaluation evidence unavailable: target absent" in html
    assert "Gate unavailable" in html


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_report_browser_workspace_replays_and_inspects_without_network(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-timeout",
        version="defective",
        discovery_cost=12,
        outcome="timed-out",
        verified=False,
        terminal_reason="run timeout exceeded",
    )
    _write_run(
        tmp_path,
        "run-evaluation",
        version="improved",
        discovery_cost=4,
        outcome="agent-abandoned",
        verified=False,
        terminal_reason="agent chose to stop",
        evaluation_failure_reason="evaluation evidence unavailable: target absent",
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "failures": [
                {
                    "run_id": "run-evaluation",
                    "error_type": "EvaluationFailure",
                    "stage": "evaluation",
                    "terminal_state": "finalized",
                    "reason": "evaluation evidence unavailable: target absent",
                    "scenario_id": "invite",
                    "application_version_id": "improved",
                    "persona_id": "persona",
                    "policy": "progressive-prominence-scent",
                    "seed": 7,
                }
            ]
        },
    )
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")
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

        overview_row = page.locator('tr[data-run-id="run-evaluation"]')
        assert (
            await overview_row.locator('[data-metric="user-actions"]').text_content()
            == "1"
        )
        assert (
            await overview_row.locator('[data-metric="observations"]').text_content()
            == "1"
        )
        assert (
            await overview_row.locator('[data-metric="discovery-cost"]').text_content()
            == "4.0"
        )
        await overview_row.click()
        assert "run=run-evaluation" in page.url
        event_ids = await page.locator(".timeline-event").evaluate_all(
            "nodes => nodes.map(node => node.dataset.eventId)"
        )
        assert event_ids == [f"event-{sequence}" for sequence in range(1, 11)]
        failure = page.locator("#run-status-banner")
        assert "agent-abandoned" in (await failure.text_content() or "")
        assert "evaluation" in (await failure.text_content() or "")
        assert "evaluation evidence unavailable: target absent" in (
            await failure.text_content() or ""
        )

        await page.locator("#play-pause").click()
        assert await page.locator("#play-pause").get_attribute("aria-pressed") == "true"
        await page.locator("#play-pause").click()
        assert (
            await page.locator("#play-pause").get_attribute("aria-pressed") == "false"
        )
        await page.locator("#step-forward").click()
        assert "Event 2 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        await page.locator("#restart-playback").click()
        assert "Event 1 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        assert "time unavailable" in (
            await page.locator("#playback-position").text_content() or ""
        )

        await page.locator('[data-event-kind="prominence-recorded"]').click()
        target = page.locator('[data-element-id="target"]').first
        await target.hover()
        panel = page.locator("#selected-element-evidence")
        assert await panel.get_attribute("data-selected-element-id") == "target"
        panel_text = await panel.text_content() or ""
        assert "0.2" in panel_text
        assert "7200" in panel_text
        assert "0.4" in panel_text
        assert "0.1" in panel_text
        assert "Occlusion fraction" in panel_text
        assert "Local contrast" in panel_text
        assert "Linked decisions" in panel_text
        assert "Linked actions and results" in panel_text
        assert "Linked findings" in panel_text

        await page.locator('[data-event-kind="action-proposed"]').click()
        event_text = await page.locator("#current-event-card").text_content() or ""
        assert "Target matches goal." in event_text
        assert "Interact With Element" in event_text
        await page.locator('[data-event-kind="model-call-recorded"]').click()
        model_text = await page.locator("#current-event-card").text_content() or ""
        assert "safe request" in model_text
        assert "safe response" in model_text
        assert "cognitive-v1" in model_text
        assert "125" in model_text
        assert "2" in model_text

        await page.locator('tr[data-run-id="run-timeout"]').click()
        timeout_text = await page.locator("#run-status-banner").text_content() or ""
        assert "timed-out" in timeout_text
        assert "run timeout exceeded" in timeout_text
        await page.set_viewport_size({"width": 390, "height": 844})
        dimensions = await page.evaluate(
            "({scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth})"
        )
        assert dimensions["scrollWidth"] <= dimensions["innerWidth"]
        assert await page.locator("#play-pause").is_visible()
        assert await page.locator("#selected-element-evidence").is_visible()
        await browser.close()

    assert not external_requests
