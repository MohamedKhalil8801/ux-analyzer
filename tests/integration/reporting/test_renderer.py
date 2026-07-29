from __future__ import annotations

import json
from pathlib import Path

from ux_analyzer.application.report import render_experiment_report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_run(
    root: Path,
    run_id: str,
    *,
    version: str,
    discovery_cost: float,
    screenshot: bytes = b"not-an-image",
) -> None:
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
            "scores": [
                {
                    "element_id": "target",
                    "raw_score": 0.2,
                    "normalized_probability": 0.3,
                    "feature_contributions": {"area": 0.1, "contrast": 0.2},
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
                "verified": True,
                "evidence_ids": ["verify-1"],
                "details": "Independent verifier passed.",
            },
        },
        {
            "sequence": 9,
            "kind": "run-terminated",
            "outcome": {"kind": "verified-success"},
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
                "verified_completion": version == "improved",
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
                    "severity": "medium",
                    "reproducibility": "model-dependent",
                    "evidence_class": "model-estimate",
                    "evidence_ids": [f"{run_id}:discovery-cost"],
                    "limitations": ["simulated benchmark evidence"],
                }
            ],
            "limitations": ["Simulated benchmark; not human satisfaction evidence."],
        },
    )


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
    assert "deterministic-fact" in html
    assert "model-estimate" in html
    assert "unsupported-human-claim" in html
    assert "Run filters" in html
    assert "Timeline" in html
    assert "Prominence contributions" in html
    assert "Scent records" in html
    assert "Decisions" in html
    assert "Actions" in html
    assert "Verification" in html
    assert "Memory" in html
    assert "Model manifests" in html
    assert "Model calls" in html
    assert "Limitations" in html
    assert "Seeded discovery cost." in html
    assert 'data-viewport-width="800"' in html
    assert "secret-token" not in html
    assert "data-testid=secret" not in html
    assert "<img src=x onerror" not in html
    assert "fetch(" not in html
    assert '<link rel="stylesheet"' not in html
    assert "<script src=" not in html


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
    assert "Directional gate" in html
    assert "All directional checks passed" in html
