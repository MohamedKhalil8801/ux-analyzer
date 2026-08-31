from __future__ import annotations

import json
from pathlib import Path

import ux_analyzer.reporting.renderer as renderer
from ux_analyzer.analysis.pagespeed import (
    PAGESPEED_FILENAME,
    PAGESPEED_SCHEMA_VERSION,
)


def _strategy_entry(**overrides) -> dict:
    entry: dict = {
        "strategy": "mobile",
        "status": "ok",
        "error": None,
        "from_cache": False,
        "fetched_at": "2026-08-27T02:57:31.788Z",
        "lighthouse_version": "12.1.0",
        "requested_url": "https://example.com/",
        "final_url": "https://example.com/",
        "main_document_url": "https://example.com/",
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
                        "wastedMs": 3200,
                        "totalBytes": 90000,
                    }
                ],
            }
        ],
        "metric_savings": [],
        "field_data": {
            "overall_category": "FAST",
            "metrics": {
                "LARGEST_CONTENTFUL_PAINT_MS": {"category": "FAST", "percentile": 901}
            },
        },
    }
    entry.update(overrides)
    return entry


def _pagespeed_payload(url_reports: list[dict]) -> dict:
    for report in url_reports:
        report.setdefault(
            "pagespeed_web_url",
            "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F",
        )
        report.setdefault(
            "pagespeed_web_fresh_url",
            "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F",
        )
        report.setdefault("pagespeed_web_saved", False)
    return {
        "schema_version": PAGESPEED_SCHEMA_VERSION,
        "urls": url_reports,
        "url_count": len(url_reports),
        "ok_strategy_count": sum(
            len(report["strategies"]) for report in url_reports
        ),
    }


def _with_link(payload: dict, link: str) -> dict:
    payload["urls"][0]["pagespeed_web_url"] = link
    return payload


class TestPagespeedWebLink:
    def test_valid_link_passes_through(self) -> None:
        assert renderer._pagespeed_web_link(
            "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        ) == "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"

    def test_non_psi_host_is_rejected(self) -> None:
        assert renderer._pagespeed_web_link(
            "https://evil.example/phish?url=https%3A%2F%2Fexample.com"
        ) is None

    def test_non_https_or_missing_is_rejected(self) -> None:
        assert renderer._pagespeed_web_link("http://pagespeed.web.dev/analysis") is None
        assert renderer._pagespeed_web_link(None) is None
        assert renderer._pagespeed_web_link("") is None

    def test_hostile_link_is_not_rendered(self, tmp_path: Path) -> None:
        payload = _pagespeed_payload(
            [{"url": "https://example.com/", "strategies": {"mobile": _strategy_entry()}}]
        )
        _write_pagespeed(
            tmp_path,
            _with_link(payload, "https://evil.example/phish"),
        )
        loaded = renderer._load_pagespeed(tmp_path)
        assert loaded["url_reports"][0]["pagespeed_web_url"] is None

    def test_hostile_fresh_link_is_not_rendered(self, tmp_path: Path) -> None:
        payload = _pagespeed_payload(
            [{"url": "https://example.com/", "strategies": {"mobile": _strategy_entry()}}]
        )
        payload["urls"][0]["pagespeed_web_fresh_url"] = "https://evil.example/x"
        _write_pagespeed(tmp_path, payload)
        loaded = renderer._load_pagespeed(tmp_path)
        assert loaded["url_reports"][0]["pagespeed_web_fresh_url"] is None

    def test_saved_report_fields_pass_through(self, tmp_path: Path) -> None:
        saved = "https://pagespeed.web.dev/analysis/https-example-com/abc123?form_factor=mobile"
        fresh = "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        payload = _pagespeed_payload(
            [{"url": "https://example.com/", "strategies": {"mobile": _strategy_entry()}}]
        )
        payload["urls"][0].update(
            {"pagespeed_web_url": saved, "pagespeed_web_fresh_url": fresh, "pagespeed_web_saved": True}
        )
        _write_pagespeed(tmp_path, payload)
        entry = renderer._load_pagespeed(tmp_path)["url_reports"][0]
        assert entry["pagespeed_web_url"] == saved
        assert entry["pagespeed_web_fresh_url"] == fresh
        assert entry["pagespeed_web_saved"] is True


def _write_pagespeed(root: Path, payload: dict) -> Path:
    path = root / PAGESPEED_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadPagespeed:
    def test_loads_ok_report(self, tmp_path: Path) -> None:
        _write_pagespeed(
            tmp_path,
            _pagespeed_payload(
                [
                    {
                        "url": "https://example.com/",
                        "strategies": {"mobile": _strategy_entry()},
                    }
                ]
            ),
        )
        loaded = renderer._load_pagespeed(tmp_path)
        assert loaded is not None
        assert loaded["url_count"] == 1
        assert loaded["ok_strategy_count"] == 1
        report = loaded["url_reports"][0]
        assert report["url"] == "https://example.com/"
        assert report["pagespeed_web_url"] == (
            "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        )
        mobile = report["strategies"]["mobile"]
        assert mobile["categories"][0]["score_percent"] == 42
        assert mobile["audits"]["failed"][0]["title"] == "Reduce unused JavaScript"
        assert mobile["audits"]["failed"][0]["failed"] is True
        assert mobile["opportunities"][0]["savings_bytes"] == 28845
        assert mobile["field_data"]["overall_category"] == "FAST"

    def test_missing_file_yields_none(self, tmp_path: Path) -> None:
        assert renderer._load_pagespeed(tmp_path) is None

    def test_wrong_schema_yields_none(self, tmp_path: Path) -> None:
        _write_pagespeed(tmp_path, {"schema_version": "other", "urls": []})
        assert renderer._load_pagespeed(tmp_path) is None

    def test_malformed_json_yields_none(self, tmp_path: Path) -> None:
        path = tmp_path / PAGESPEED_FILENAME
        path.write_text("{not json", encoding="utf-8")
        assert renderer._load_pagespeed(tmp_path) is None

    def test_error_strategy_is_kept_with_error_text(self, tmp_path: Path) -> None:
        _write_pagespeed(
            tmp_path,
            _pagespeed_payload(
                [
                    {
                        "url": "https://down.example/",
                        "strategies": {
                            "mobile": {
                                "strategy": "mobile",
                                "status": "error",
                                "error": "PagespeedApiError: API request failed (HTTP 429)",
                            }
                        },
                    }
                ]
            ),
        )
        loaded = renderer._load_pagespeed(tmp_path)
        entry = loaded["url_reports"][0]["strategies"]["mobile"]
        assert entry["status"] == "error"
        assert "HTTP 429" in entry["error"]

    def test_junk_fields_are_dropped(self, tmp_path: Path) -> None:
        _write_pagespeed(
            tmp_path,
            _pagespeed_payload(
                [
                    {
                        "url": "https://example.com/",
                        "error": "url-level failure",
                        "strategies": {
                            "mobile": _strategy_entry(
                                audits={
                                    "failed": [
                                        {
                                            "id": "ok-audit",
                                            "title": "OK audit",
                                            "score": 0.5,
                                            "score_percent": 50,
                                            "score_display_mode": "numeric",
                                            "display_value": None,
                                            "description": None,
                                            "failed": True,
                                        },
                                        {"nonsense": True},
                                    ]
                                }
                            )
                        },
                    }
                ]
            ),
        )
        loaded = renderer._load_pagespeed(tmp_path)
        failed = loaded["url_reports"][0]["strategies"]["mobile"]["audits"]["failed"]
        assert [row["id"] for row in failed] == ["ok-audit"]


class TestSavedReportScores:
    def test_scores_pass_through_with_capture_date(self, tmp_path: Path) -> None:
        payload = _pagespeed_payload(
            [{"url": "https://example.com/", "strategies": {"mobile": _strategy_entry()}}]
        )
        payload["urls"][0]["saved_report_scores"] = {"mobile": 36, "desktop": 63}
        payload["urls"][0]["saved_report_captured_at"] = "2026-08-30T12:00:00Z"
        _write_pagespeed(tmp_path, payload)
        entry = renderer._load_pagespeed(tmp_path)["url_reports"][0]
        assert entry["saved_report_scores"] == {"mobile": 36, "desktop": 63}
        assert entry["saved_report_captured_at"] == "2026-08-30T12:00:00Z"

    def test_missing_scores_default_to_empty(self, tmp_path: Path) -> None:
        _write_pagespeed(
            tmp_path,
            _pagespeed_payload(
                [{"url": "https://example.com/", "strategies": {"mobile": _strategy_entry()}}]
            ),
        )
        entry = renderer._load_pagespeed(tmp_path)["url_reports"][0]
        assert entry["saved_report_scores"] == {}
        assert entry["saved_report_captured_at"] is None

    def test_invalid_scores_are_dropped(self, tmp_path: Path) -> None:
        payload = _pagespeed_payload(
            [{"url": "https://example.com/", "strategies": {"mobile": _strategy_entry()}}]
        )
        payload["urls"][0]["saved_report_scores"] = {
            "mobile": 200,
            "desktop": "63",
            "pwa": True,
            "other": 1.5,
        }
        _write_pagespeed(tmp_path, payload)
        entry = renderer._load_pagespeed(tmp_path)["url_reports"][0]
        assert entry["saved_report_scores"] == {}


class TestSavedReportReplica:
    def _payload(self) -> dict:
        return {
            "url_reports": [
                {
                    "url": "https://example.com/",
                    "error": None,
                    "pagespeed_web_saved": False,
                    "saved_report_scores": {},
                    "saved_report_captured_at": None,
                    "strategies": {
                        "mobile": {
                            "strategy": "mobile",
                            "status": "ok",
                            "error": None,
                            "error_code": None,
                            "fetched_at": "2026-08-30T14:15:50Z",
                            "analysis_timestamp": "2026-08-30T14:15:44Z",
                            "lighthouse_version": "12.1.0",
                            "from_cache": False,
                            "final_url": "https://example.com/",
                            "categories": [
                                {
                                    "id": "performance",
                                    "title": "Performance",
                                    "score": 0.42,
                                    "score_percent": 42,
                                    "display_value": None,
                                    "audit_refs": 47,
                                }
                            ],
                            "audits": {
                                "failed": [
                                    {
                                        "id": "total-blocking-time",
                                        "title": "Total Blocking Time",
                                        "score": 0.0,
                                        "score_percent": 0,
                                        "score_display_mode": "numeric",
                                        "display_value": "10 ms",
                                        "description": None,
                                        "failed": True,
                                    }
                                ],
                                "passed": [
                                    {
                                        "id": "first-contentful-paint",
                                        "title": "First Contentful Paint",
                                        "score": 0.28,
                                        "score_percent": 28,
                                        "score_display_mode": "numeric",
                                        "display_value": "3.8 s",
                                        "description": None,
                                        "failed": False,
                                    }
                                ],
                                "not_applicable": [],
                                "manual": [],
                                "informative": [],
                                "error": [],
                                "totals": {},
                            },
                            "opportunities": [],
                            "metric_savings": [],
                            "field_data": None,
                        }
                    },
                }
            ]
        }

    def test_core_metrics_are_collected_in_lighthouse_order(self) -> None:
        strategy = self._payload()["url_reports"][0]["strategies"]["mobile"]
        metrics = renderer._pagespeed_core_metrics(strategy)
        assert [m["id"] for m in metrics][:2] == [
            "first-contentful-paint",
            "total-blocking-time",
        ]

    def test_replica_html_contains_same_scores(self) -> None:
        html = renderer._render_pagespeed_saved_html(self._payload())
        assert "recorded analysis replica" in html
        assert 'psi-gauge-ring">42</span>' in html
        assert "Total Blocking Time" in html
        assert "10 ms" in html

    def test_replica_is_written_beside_the_report(self, tmp_path: Path) -> None:
        experiment = {"pagespeed": self._payload()}
        destination = renderer._publish_pagespeed_saved_report(
            experiment, tmp_path
        )
        assert destination == tmp_path / "pagespeed-report.html"
        assert destination.is_file()

    def test_no_pagespeed_yields_no_replica(self, tmp_path: Path) -> None:
        assert renderer._publish_pagespeed_saved_report({}, tmp_path) is None
        assert not (tmp_path / "pagespeed-report.html").exists()
