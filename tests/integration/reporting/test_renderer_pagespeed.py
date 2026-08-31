from __future__ import annotations

import json
import re
from pathlib import Path

from ux_analyzer.analysis.pagespeed import PAGESPEED_SCHEMA_VERSION
from ux_analyzer.reporting.renderer import render_experiment_report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _minimal_run_bundle(root: Path, run_id: str) -> None:
    run = root / "runs" / run_id
    _write_json(
        run / "manifest.json",
        {
            "run_id": run_id,
            "seed": 7,
            "model_trial": 1,
            "prominence_provider_id": "heuristic",
            "config_digest": "config-sha",
        },
    )
    _write_json(
        run / "result.json",
        {
            "run_id": run_id,
            "seed": 7,
            "model_trial": 1,
            "prominence_provider_id": "heuristic",
            "config_digest": "config-sha",
            "scenario_id": "scenario-1",
            "application_version_id": "app-live",
            "persona_id": "persona-1",
            "policy": "full-list",
            "state": {"events": [], "spec": {}},
        },
    )
    (run / "timeline.jsonl").write_text("", encoding="utf-8")
    (run / "artifacts").mkdir(parents=True, exist_ok=True)


def _pagespeed_payload() -> dict:
    return {
        "schema_version": PAGESPEED_SCHEMA_VERSION,
        "urls": [
            {
                "url": "https://example.com/",
                "pagespeed_web_url": (
                    "https://pagespeed.web.dev/analysis/https-example-com/abc123"
                    "?form_factor=mobile"
                ),
                "pagespeed_web_fresh_url": (
                    "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
                ),
                "pagespeed_web_saved": True,
                "strategies": {
                    "mobile": {
                        "strategy": "mobile",
                        "status": "ok",
                        "error": None,
                        "from_cache": False,
                        "fetched_at": "2026-08-27T02:57:31.788Z",
                        "requested_url": "https://example.com/",
                        "final_url": "https://example.com/",
                        "lighthouse_version": "12.1.0",
                        "user_agent": "Mozilla/5.0",
                        "categories": [
                            {
                                "id": "performance",
                                "title": "Performance",
                                "score": 0.42,
                                "score_percent": 42,
                                "display_value": "42 s",
                                "audit_refs": 47,
                            }
                        ],
                        "audits": {
                            "failed": [
                                {
                                    "id": "unused-javascript",
                                    "title": "Reduce unused JavaScript",
                                    "score": 0.1,
                                    "score_percent": 10,
                                    "score_display_mode": "numeric",
                                    "display_value": "10 s",
                                    "description": "Reduce unused JavaScript.",
                                    "failed": True,
                                }
                            ],
                            "passed": [
                                {
                                    "id": "font-display",
                                    "title": "Fonts with font-display",
                                    "score": 1.0,
                                    "score_percent": 100,
                                    "score_display_mode": "binary",
                                    "display_value": None,
                                    "description": None,
                                    "failed": False,
                                }
                            ],
                            "not_applicable": [],
                            "manual": [],
                            "informative": [],
                            "error": [],
                            "totals": {
                                "failed": 1,
                                "passed": 1,
                                "not_applicable": 0,
                                "manual": 0,
                                "informative": 0,
                                "error": 0,
                            },
                        },
                        "opportunities": [
                            {
                                "id": "unused-javascript",
                                "title": "Reduce unused JavaScript",
                                "score": 0.1,
                                "score_display_mode": "numeric",
                                "display_value": "10 s",
                                "savings_ms": 3200,
                                "savings_bytes": 28845,
                                "items": [
                                    {
                                        "url": "https://example.com/app.js",
                                        "wastedBytes": 28845,
                                        "totalBytes": 90000,
                                    }
                                ],
                            }
                        ],
                        "metric_savings": [],
                        "field_data": {
                            "overall_category": "FAST",
                            "metrics": {
                                "LARGEST_CONTENTFUL_PAINT_MS": {
                                    "category": "FAST",
                                    "percentile": 901,
                                }
                            },
                        },
                    }
                },
            }
        ],
        "url_count": 1,
        "ok_strategy_count": 1,
    }


def test_render_experiment_report_embeds_pagespeed_section(tmp_path: Path) -> None:
    _minimal_run_bundle(tmp_path, "run-1")
    _write_json(tmp_path / "pagespeed.json", _pagespeed_payload())

    output = tmp_path / "report.html"
    rendered = render_experiment_report(tmp_path, output)

    assert rendered == output
    html = output.read_text(encoding="utf-8")
    assert 'id="pagespeed"' in html
    assert "PageSpeed Insights" in html
    assert "Performance" in html
    assert "42" in html
    assert "Reduce unused JavaScript" in html
    assert "~3200 ms" in html
    assert "~28845 bytes" in html
    assert "https://example.com/app.js" in html
    assert "FAST" in html
    assert "Failed audits (1)" in html
    assert 'href="pagespeed-report.html"' in html
    assert "View saved report" in html
    assert (
        'href="https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"'
        in html
    )
    assert "Re-run on pagespeed.web.dev" in html

    saved_report = tmp_path / "pagespeed-report.html"
    assert saved_report.is_file()
    replica = saved_report.read_text(encoding="utf-8")
    assert "recorded analysis replica" in replica
    assert "42" in replica
    assert "Performance" in replica
    assert "Reduce unused JavaScript" in replica
    assert "FAST" in replica


def test_saved_replica_scores_match_main_report(tmp_path: Path) -> None:
    _minimal_run_bundle(tmp_path, "run-1")
    _write_json(tmp_path / "pagespeed.json", _pagespeed_payload())

    render_experiment_report(tmp_path, tmp_path / "report.html")

    main = (tmp_path / "report.html").read_text(encoding="utf-8")
    replica = (tmp_path / "pagespeed-report.html").read_text(encoding="utf-8")
    main_scores = re.findall(
        r'psi-category-title">Performance</span>\s*'
        r'<span class="psi-category-score">(\d+)</span>',
        main,
    )
    replica_scores = re.findall(
        r'psi-gauge-ring">(\d+)</span>\s*<span class="psi-gauge-label">Performance',
        replica,
    )
    assert main_scores == replica_scores == ["42"]


def test_render_experiment_report_escapes_html_markup_in_audit_titles(
    tmp_path: Path,
) -> None:
    _minimal_run_bundle(tmp_path, "run-1")
    payload = _pagespeed_payload()
    strategy = payload["urls"][0]["strategies"]["mobile"]
    strategy["audits"]["passed"].append(
        {
            "id": "document-title",
            "title": "Document has a `<title>` element",
            "score": 1.0,
            "score_percent": 100,
            "score_display_mode": "binary",
            "display_value": None,
            "description": None,
            "failed": False,
        }
    )
    strategy["audits"]["totals"]["passed"] = 2
    _write_json(tmp_path / "pagespeed.json", payload)

    output = tmp_path / "report.html"
    rendered = render_experiment_report(tmp_path, output)

    assert rendered == output
    html = output.read_text(encoding="utf-8")
    assert html.count("<title>") == 1
    assert "Document has a `&lt;title&gt;` element" in html
