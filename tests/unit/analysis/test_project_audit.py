from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import ux_analyzer.cli as cli
from ux_analyzer.analysis.project_audit import (
    AUDIT_FILENAME,
    AUDIT_SCHEMA_VERSION,
    audit_urls,
    audit_urls_sync,
)


def _result_with_start_url(url: str | None) -> object:
    version = SimpleNamespace(start_url=url)
    spec = SimpleNamespace(application_version=version)
    state = SimpleNamespace(spec=spec)
    return SimpleNamespace(state=state)


@pytest.mark.asyncio
async def test_audit_urls_deduplicates_and_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    async def fake_audit_url(url: str) -> dict[str, Any]:
        seen.append(url)
        return {
            "url": url,
            "counts": {"GEO": 1},
            "total": 1,
            "issues": [
                {
                    "category": "GEO",
                    "check_id": "robots_txt",
                    "title": "robots.txt not found",
                    "severity": "critical",
                    "evidence": {"status_code": 404},
                }
            ],
        }

    monkeypatch.setattr(
        "ux_analyzer.analysis.project_audit.audit_url", fake_audit_url
    )
    report = await audit_urls(
        ("https://a.example/", "https://a.example/", "", "https://b.example/")
    )
    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report["total_issues"] == 2
    assert [entry["url"] for entry in report["urls"]] == [
        "https://a.example/",
        "https://b.example/",
    ]
    assert seen == ["https://a.example/", "https://b.example/"]
    assert report["errors"] == []


@pytest.mark.asyncio
async def test_audit_urls_records_errors_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_audit_url(url: str) -> dict[str, Any]:
        raise RuntimeError("network down")

    monkeypatch.setattr(
        "ux_analyzer.analysis.project_audit.audit_url", failing_audit_url
    )
    report = await audit_urls(("https://down.example/",))
    assert report["total_issues"] == 0
    assert report["urls"] == []
    assert len(report["errors"]) == 1
    assert "RuntimeError" in report["errors"][0]["error"]


def test_audit_urls_sync_runs_without_running_loop() -> None:
    report = audit_urls_sync(())
    assert report["total_issues"] == 0
    assert report["urls"] == []


def test_ux_audit_start_urls_collected_and_deduplicated() -> None:
    results = [
        _result_with_start_url("https://a.example/"),
        _result_with_start_url(None),
        _result_with_start_url("https://a.example/"),
        SimpleNamespace(),
    ]
    urls = cli._ux_audit_start_urls(results)
    assert urls == ("https://a.example/",)


def test_write_ux_audit_persists_report(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_sync(
        urls: object, *, capture_hook: object | None = None
    ) -> dict[str, Any]:
        del capture_hook
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "total_issues": 1,
            "urls": [
                {
                    "url": "https://a.example/",
                    "total": 1,
                    "issues": [
                        {
                            "category": "GEO",
                            "check_id": "robots_txt",
                            "title": "robots.txt not found",
                            "severity": "critical",
                            "evidence": {},
                        }
                    ],
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr(cli, "_ux_audit_sync", fake_sync)
    destination = cli._write_ux_audit(
        tmp_path,
        (_result_with_start_url("https://a.example/"),),
    )
    assert destination is not None and destination.name == AUDIT_FILENAME
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["total_issues"] == 1
    assert payload["urls"][0]["issues"][0]["title"] == "robots.txt not found"


def test_write_ux_audit_persists_page_capture_sidecar(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared audit pass leaves a page-capture.json beside the audit."""

    import io as _io

    from PIL import Image

    from ux_analyzer.analysis.project_audit import CaptureMaterials

    buffer = _io.BytesIO()
    Image.new("RGB", (8, 8), (10, 10, 10)).save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    def fake_sync(
        urls: object, *, capture_hook: object | None = None
    ) -> dict[str, Any]:
        if capture_hook is not None:
            capture_hook(
                "https://a.example/",
                CaptureMaterials(
                    url="https://a.example/",
                    png_bytes=png_bytes,
                    title="Sidecar",
                    document_height=8,
                    page=None,
                ),
            )
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "total_issues": 0,
            "urls": [],
            "errors": [],
        }

    monkeypatch.setattr(cli, "_ux_audit_sync", fake_sync)
    destination = cli._write_ux_audit(
        tmp_path,
        (_result_with_start_url("https://a.example/"),),
    )
    assert destination is not None
    sidecar = tmp_path / "page-capture.json"
    assert sidecar.exists()
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["schema"] == "page-capture-v2"
    assert len(document["pages"]) == 1
    page = document["pages"][0]
    assert page["url"] == "https://a.example/"
    assert page["title"] == "Sidecar"
    assert page["segments"][0]["data_url"].startswith("data:image/jpeg;base64,")


def test_write_ux_audit_tolerates_capture_errors(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture failures never block the audit or leave a partial sidecar."""

    def fake_sync(
        urls: object, *, capture_hook: object | None = None
    ) -> dict[str, Any]:
        if capture_hook is not None:
            capture_hook("https://a.example/", object())  # unusable materials
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "total_issues": 0,
            "urls": [],
            "errors": [],
        }

    monkeypatch.setattr(cli, "_ux_audit_sync", fake_sync)
    destination = cli._write_ux_audit(
        tmp_path,
        (_result_with_start_url("https://a.example/"),),
    )
    assert destination is not None
    assert not (tmp_path / "page-capture.json").exists()


def test_write_ux_audit_skips_when_no_urls(tmp_path: Any) -> None:
    assert cli._write_ux_audit(tmp_path, (SimpleNamespace(),)) is None
    assert not (tmp_path / AUDIT_FILENAME).exists()


def test_write_ux_audit_survives_audit_failure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_sync(urls: object) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_ux_audit_sync", failing_sync)
    destination = cli._write_ux_audit(
        tmp_path,
        (_result_with_start_url("https://a.example/"),),
    )
    assert destination is None
    assert not (tmp_path / AUDIT_FILENAME).exists()
