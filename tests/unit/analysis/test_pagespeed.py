from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from ux_analyzer.analysis.pagespeed import (
    CACHE_DIRNAME,
    PAGESPEED_SCHEMA_VERSION,
    PagespeedApiError,
    PagespeedCache,
    WebLinksCache,
    _normalize_saved_link,
    aggregate_categories,
    build_url_report,
    enrich_pagespeed_web_links,
    extract_audits,
    extract_field_data,
    extract_opportunities,
    extract_pagespeed_web_saved_scores,
    fetch_pagespeed,
    pagespeed_api_key,
    pagespeed_report,
    pagespeed_report_sync,
    pagespeed_web_url,
    resolve_pagespeed_web_saved_link,
    resolve_pagespeed_web_saved_report,
)


def _audit(
    audit_id: str,
    *,
    mode: str = "binary",
    score: float | None = 1.0,
    display_value: str | None = None,
    details: dict | None = None,
) -> dict:
    row: dict = {
        "id": audit_id,
        "title": audit_id.replace("-", " ").title(),
        "description": f"{audit_id} description",
        "score": score,
        "scoreDisplayMode": mode,
    }
    if display_value is not None:
        row["displayValue"] = display_value
    if details is not None:
        row["details"] = details
    return row


def _lighthouse(audits: dict, categories: dict | None = None) -> dict:
    return {
        "lighthouseResult": {
            "requestedUrl": "https://example.com/",
            "finalUrl": "https://example.com/",
            "lighthouseVersion": "12.1.0",
            "fetchTime": "2026-08-27T02:57:31.788Z",
            "audits": audits,
            "categories": categories or {
                "performance": {
                    "id": "performance",
                    "title": "Performance",
                    "score": 0.42,
                    "displayValue": "42 s",
                    "auditRefs": [{}] * 47,
                }
            },
        }
    }


def _opportunity_details(savings_ms: float, savings_bytes: float, items: list[dict]) -> dict:
    details: dict = {"type": "opportunity"}
    if savings_ms is not None:
        details["overallSavingsMs"] = savings_ms
    if savings_bytes is not None:
        details["overallSavingsBytes"] = savings_bytes
    if items is not None:
        details["items"] = items
    return details


class _FakeAsyncClient:
    """httpx.AsyncClient stand-in returning queued responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    async def get(self, url: str) -> httpx.Response:
        self.urls.append(url)
        if not self.responses:
            return httpx.Response(503, json={})
        return self.responses.pop(0)

    async def aclose(self) -> None:
        pass


class TestApiKey:
    def test_reads_psi_api_key_variant(self) -> None:
        assert pagespeed_api_key({"PSI_API_Key": "AIzaSyabc123"}) == "AIzaSyabc123"

    def test_reads_googapi_variant(self) -> None:
        assert pagespeed_api_key({"GOOGLE_API_KEY": "AIzaSyabc123"}) == "AIzaSyabc123"

    def test_rejects_short_or_spaced_values(self) -> None:
        assert pagespeed_api_key({"PSI_API_KEY": "x"}) is None
        assert pagespeed_api_key({"PSI_API_KEY": "has space here"}) is None

    def test_empty_environment_returns_none(self) -> None:
        assert pagespeed_api_key({}) is None


class TestFetchPagespeed:
    @pytest.fixture(autouse=True)
    def _zero_retry_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retry sequencing is under test, not wall-clock backoff."""
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed._RETRY_DELAYS", (0.0, 0.0, 0.0)
        )

    async def test_returns_payload_on_200(self) -> None:
        payload = {"lighthouseResult": {"fetchTime": "now"}}
        client = _FakeAsyncClient([httpx.Response(200, json=payload)])
        result = await fetch_pagespeed("https://example.com", "mobile", client=client)
        assert result == payload
        assert client.urls[0].startswith(
            "https://www.googleapis.com/pagespeedonline/v5/runPagespeed?"
        )
        assert "url=https%3A%2F%2Fexample.com" in client.urls[0]
        assert "strategy=mobile" in client.urls[0]
        assert "category=performance" in client.urls[0]

    async def test_appends_key_when_provided(self) -> None:
        client = _FakeAsyncClient([httpx.Response(200, json={"ok": True})])
        await fetch_pagespeed(
            "https://example.com", "mobile", key="AIzaSyabc123", client=client
        )
        assert "key=AIzaSyabc123" in client.urls[0]

    async def test_retries_429_then_succeeds(self) -> None:
        client = _FakeAsyncClient(
            [
                httpx.Response(429, json={"error": {"code": 429, "message": "quota"}}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        result = await fetch_pagespeed(
            "https://example.com", "mobile", client=client, retries=2
        )
        assert result == {"ok": True}

    async def test_raises_after_exhausted_retries(self) -> None:
        client = _FakeAsyncClient([httpx.Response(429, json={})])
        with pytest.raises(PagespeedApiError) as exc:
            await fetch_pagespeed("https://example.com", "mobile", client=client, retries=0)
        assert exc.value.status_code == 429

    async def test_400_raises_without_retry(self) -> None:
        client = _FakeAsyncClient(
            [httpx.Response(400, json={"error": {"code": 400, "message": "bad url"}})]
        )
        with pytest.raises(PagespeedApiError) as exc:
            await fetch_pagespeed("https://example.com", "mobile", client=client, retries=3)
        assert exc.value.status_code == 400
        assert "bad url" in (exc.value.error_code or "")
        assert len(client.responses) == 0

    async def test_non_json_error_body_retries_and_reports_status(self) -> None:
        client = _FakeAsyncClient(
            [
                httpx.Response(
                    503, content=b"<html><body>Bad Gateway</body></html>"
                ),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        result = await fetch_pagespeed(
            "https://example.com", "mobile", client=client, retries=2
        )
        assert result == {"ok": True}

    async def test_non_json_error_body_exhausts_retries_with_pagespeed_error(self) -> None:
        client = _FakeAsyncClient(
            [httpx.Response(503, content=b"<html>down</html>")]
        )
        with pytest.raises(PagespeedApiError) as exc:
            await fetch_pagespeed("https://example.com", "mobile", client=client, retries=1)
        assert exc.value.status_code == 503
        assert exc.value.error_code is None

    async def test_deeply_nested_success_body_becomes_pagespeed_error(self) -> None:
        client = _FakeAsyncClient(
            [httpx.Response(200, content=b"[" * 20000 + b"]" * 20000)]
        )
        with pytest.raises(PagespeedApiError):
            await fetch_pagespeed("https://example.com", "mobile", client=client)

    async def test_deeply_nested_error_body_does_not_raise(self) -> None:
        client = _FakeAsyncClient(
            [httpx.Response(503, content=b"[" * 20000 + b"]" * 20000)]
        )
        with pytest.raises(PagespeedApiError) as exc:
            await fetch_pagespeed("https://example.com", "mobile", client=client, retries=0)
        assert exc.value.status_code == 503
        assert exc.value.error_code is None

    async def test_network_failure_backs_off_and_raises_pagespeed_error(self) -> None:
        class _FailingClient(_FakeAsyncClient):
            def __init__(self) -> None:
                super().__init__([])
                self.attempts = 0
                self.backoffs: list[float] = []

            async def get(self, url: str) -> httpx.Response:
                self.attempts += 1
                raise httpx.ConnectTimeout("boom", request=httpx.Request("GET", url))

        client = _FailingClient()
        with pytest.raises(PagespeedApiError) as exc:
            await fetch_pagespeed(
                "https://example.com", "mobile", client=client, retries=2
            )
        assert client.attempts == 3
        assert exc.value.status_code is None
        assert "network failure" in exc.value.reason

    async def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError):
            await fetch_pagespeed("https://example.com", "tablet")

    async def test_rejects_empty_url(self) -> None:
        with pytest.raises(ValueError):
            await fetch_pagespeed("  ", "mobile")


class TestCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cache = PagespeedCache(tmp_path)
        assert cache.load("https://example.com", "mobile") is None
        path = cache.store("https://example.com", "mobile", {"score": 0.42})
        assert path.is_file()
        assert cache.load("https://example.com", "mobile") == {"score": 0.42}
        assert cache.load("https://example.com", "desktop") is None
        assert cache.load("https://other.example", "mobile") is None

    def test_refuses_oversized_store_and_never_loads_it(self, tmp_path: Path) -> None:
        cache = PagespeedCache(tmp_path)
        with pytest.raises(OSError):
            cache.store("https://example.com", "mobile", {"big": "x" * (9 * 1024 * 1024)})
        assert cache.load("https://example.com", "mobile") is None

    def test_refuses_symlinked_cache_entry(self, tmp_path: Path) -> None:
        cache = PagespeedCache(tmp_path)
        secret = tmp_path / "secret.json"
        secret.write_text(json.dumps({"marker": "OUTSIDE"}), encoding="utf-8")
        link = cache._path_for("https://example.com", "mobile")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(secret)
        assert cache.load("https://example.com", "mobile") is None

    def test_refuses_link_on_cache_directory(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        cache = PagespeedCache(tmp_path)
        outside = tmp_path.parent / f"outside-{tmp_path.name}"
        outside.mkdir(exist_ok=True)
        (outside / "secret.json").write_text(
            json.dumps({"marker": "SMUGGLED"}), encoding="utf-8"
        )
        cache_dir = tmp_path / CACHE_DIRNAME
        if sys.platform == "win32":
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(cache_dir), str(outside)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.skip("cannot create directory junction")
        else:
            cache_dir.symlink_to(outside, target_is_directory=True)
        assert cache.load("https://example.com", "mobile") is None
        with pytest.raises(OSError):
            cache.store("https://example.com", "mobile", {"smuggle": True})

    def test_refuses_symlink_at_temporary_path(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        cache = PagespeedCache(tmp_path)
        victim = tmp_path.parent / f"victim-{tmp_path.name}"
        victim.write_text("ORIGINAL", encoding="utf-8")
        digest_path = cache._path_for("https://example.com", "mobile")
        temp_path = digest_path.with_name(f".{digest_path.name}.fixed.tmp")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            result = subprocess.run(
                ["cmd", "/c", "mklink", str(temp_path), str(victim)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.skip("cannot create file symlink on this system")
        else:
            temp_path.symlink_to(victim)
        cache._temporary_path = lambda path: temp_path  # type: ignore[method-assign]
        with pytest.raises(OSError):
            cache.store("https://example.com", "mobile", {"smuggle": True})
        assert victim.read_text(encoding="utf-8") == "ORIGINAL"

    def test_concurrent_stores_to_same_key_do_not_collide(self, tmp_path: Path) -> None:
        import asyncio

        cache = PagespeedCache(tmp_path)

        async def worker() -> None:
            for _ in range(3):
                await asyncio.to_thread(
                    cache.store, "https://example.com", "mobile", {"n": 1}
                )

        async def main() -> None:
            await asyncio.gather(*(worker() for _ in range(4)))

        asyncio.run(main())
        assert cache.load("https://example.com", "mobile") == {"n": 1}

    def test_refuses_root_that_is_a_junction(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        outside = tmp_path.parent / f"root-target-{tmp_path.name}"
        outside.mkdir(exist_ok=True)
        root_link = tmp_path / "cache-root"
        if sys.platform == "win32":
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(root_link), str(outside)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.skip("cannot create directory junction")
        else:
            root_link.symlink_to(outside, target_is_directory=True)
        cache = PagespeedCache(root_link)
        assert cache.load("https://example.com", "mobile") is None
        with pytest.raises(OSError):
            cache.store("https://example.com", "mobile", {"smuggle": True})
        assert not (outside / CACHE_DIRNAME).exists()


class TestAggregateCategories:
    def test_scores_and_percents_from_source(self) -> None:
        categories = aggregate_categories(_lighthouse({}))
        assert categories == [
            {
                "id": "performance",
                "title": "Performance",
                "score": 0.42,
                "score_percent": 42,
                "display_value": "42 s",
                "audit_refs": 47,
            }
        ]

    def test_missing_score_keeps_none(self) -> None:
        raw = _lighthouse({})
        raw["lighthouseResult"]["categories"]["performance"]["score"] = None
        row = aggregate_categories(raw)[0]
        assert row["score"] is None
        assert row["score_percent"] is None

    def test_absent_categories_yield_empty_list(self) -> None:
        assert aggregate_categories({"lighthouseResult": {}}) == []


class TestExtractAudits:
    def test_buckets_by_mode_and_threshold(self) -> None:
        audits = {
            "passed-binary": _audit("passed-binary", mode="binary", score=1.0),
            "passed-numeric": _audit("passed-numeric", mode="numeric", score=0.95),
            "failed-numeric": _audit("failed-numeric", mode="numeric", score=0.42),
            "failed-binary": _audit("failed-binary", mode="binary", score=0.0),
            "not-applicable": _audit(
                "not-applicable", mode="notApplicable", score=None
            ),
            "manual": _audit("manual", mode="manual", score=None),
            "informative": _audit("informative", mode="informative", score=0.5),
            "error-mode": _audit("error-mode", mode="error", score=None),
            "metric-savings": _audit(
                "metric-savings", mode="metricSavings", score=0.3
            ),
        }
        buckets = extract_audits(_lighthouse(audits))
        assert [row["id"] for row in buckets["passed"]] == [
            "passed-binary",
            "passed-numeric",
        ]
        assert [row["id"] for row in buckets["failed"]] == [
            "failed-binary",
            "failed-numeric",
            "metric-savings",
        ]
        assert [row["id"] for row in buckets["not_applicable"]] == ["not-applicable"]
        assert [row["id"] for row in buckets["manual"]] == ["manual"]
        assert [row["id"] for row in buckets["informative"]] == ["informative"]
        assert [row["id"] for row in buckets["error"]] == ["error-mode"]
        row = buckets["failed"][0]
        assert row["score"] == 0.0
        assert row["failed"] is True
        assert row["display_value"] is None
        assert row["description"] == "failed-binary description"

    def test_no_audits_returns_empty_buckets(self) -> None:
        buckets = extract_audits(_lighthouse({}))
        assert all(len(rows) == 0 for rows in buckets.values())


class TestExtractOpportunities:
    def test_opportunity_with_savings_and_items(self) -> None:
        audits = {
            "unused-javascript": _audit(
                "unused-javascript",
                mode="numeric",
                score=0.1,
                display_value="10 s",
                details=_opportunity_details(
                    3200,
                    28845,
                    [{"url": "https://example.com/app.js", "wastedBytes": 28845}],
                ),
            )
        }
        result = extract_opportunities(_lighthouse(audits))
        assert len(result["opportunities"]) == 1
        row = result["opportunities"][0]
        assert row["id"] == "unused-javascript"
        assert row["savings_ms"] == 3200
        assert row["savings_bytes"] == 28845
        assert row["display_value"] == "10 s"
        assert row["items"] == [{"url": "https://example.com/app.js", "wastedBytes": 28845}]

    def test_items_are_filtered_to_known_keys(self) -> None:
        audits = {
            "opp": _audit(
                "opp",
                details=_opportunity_details(
                    None,
                    None,
                    [
                        {
                            "url": "https://example.com/a.js",
                            "wastedBytes": 10,
                            "totalBytes": 20,
                            "nonsense": "dropped",
                            "wastedMs": "not-a-number",
                        }
                    ],
                ),
            )
        }
        row = extract_opportunities(_lighthouse(audits))["opportunities"][0]
        assert row["items"] == [{"url": "https://example.com/a.js", "wastedBytes": 10, "totalBytes": 20}]

    def test_non_opportunity_without_savings_is_dropped(self) -> None:
        audits = {
            "diagnostic": _audit(
                "diagnostic", details={"type": "diagnostic", "items": [{"url": "x"}]}
            )
        }
        result = extract_opportunities(_lighthouse(audits))
        assert result["opportunities"] == []
        assert result["metric_savings"] == []

    def test_metric_savings_audits_collected(self) -> None:
        audits = {
            "mainthread-work-breakdown": _audit(
                "mainthread-work-breakdown",
                mode="metricSavings",
                score=0.0,
                display_value="39.8 s",
                details={"type": "metricSavings", "overallSavingsMs": 4300},
            )
        }
        result = extract_opportunities(_lighthouse(audits))
        assert result["metric_savings"] == [
            {
                "id": "mainthread-work-breakdown",
                "title": "Mainthread Work Breakdown",
                "score": 0.0,
                "score_display_mode": "metricSavings",
                "display_value": "39.8 s",
                "savings_ms": 4300,
                "savings_bytes": None,
                "items": [],
            }
        ]

    def test_opportunities_sorted_by_savings_descending(self) -> None:
        audits = {
            "small": _audit(
                "small",
                details=_opportunity_details(100, None, []),
            ),
            "large": _audit(
                "large",
                details=_opportunity_details(5000, None, []),
            ),
        }
        ids = [row["id"] for row in extract_opportunities(_lighthouse(audits))["opportunities"]]
        assert ids == ["large", "small"]


class TestExtractFieldData:
    def test_field_data_passthrough(self) -> None:
        raw = _lighthouse({})
        raw["loadingExperience"] = {
            "overall_category": "FAST",
            "metrics": {
                "LARGEST_CONTENTFUL_PAINT_MS": {"category": "FAST", "percentile": 901}
            },
        }
        result = extract_field_data(raw)
        assert result == {
            "overall_category": "FAST",
            "metrics": {"LARGEST_CONTENTFUL_PAINT_MS": {"category": "FAST", "percentile": 901}},
        }

    def test_absent_field_data_yields_none(self) -> None:
        assert extract_field_data(_lighthouse({})) is None


class TestPagespeedWebUrl:
    def test_builds_deep_link_with_encoded_url(self) -> None:
        assert pagespeed_web_url("https://example.com/") == (
            "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        )

    def test_encodes_query_parameters(self) -> None:
        url = pagespeed_web_url("https://example.com/?a=b&c=d")
        assert "?url=https%3A%2F%2Fexample.com%2F%3Fa%3Db%26c%3Dd" in url

    def test_strips_surrounding_whitespace(self) -> None:
        assert pagespeed_web_url("  https://example.com  ").endswith(
            "%2F%2Fexample.com"
        )


class _FakePage:
    def __init__(self, urls: list[str], bodies: list[str]) -> None:
        self._urls = urls
        self._bodies = bodies
        self._index = 0

    def goto(self, *args, **kwargs) -> None:
        pass

    def click(self, *args, **kwargs) -> None:
        raise RuntimeError("no consent dialog")

    def wait_for_timeout(self, milliseconds: int) -> None:
        self._index += 1

    @property
    def url(self) -> str:
        return self._urls[min(self._index, len(self._urls) - 1)]

    def inner_text(self, selector: str) -> str:
        return self._bodies[min(self._index, len(self._bodies) - 1)]


class _FakeContext:
    def __init__(self, urls: list[str], bodies: list[str]) -> None:
        self._urls = urls
        self._bodies = bodies

    def new_page(self) -> _FakePage:
        return _FakePage(self._urls, self._bodies)


class _FakeBrowser:
    def __init__(self, urls: list[str], bodies: list[str]) -> None:
        self._urls = urls
        self._bodies = bodies
        self.closed = False

    def new_context(self) -> _FakeContext:
        return _FakeContext(self._urls, self._bodies)

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, urls: list[str], bodies: list[str]) -> None:
        self._urls = urls
        self._bodies = bodies
        self._browser: _FakeBrowser | None = None

    def launch(self) -> _FakeBrowser:
        self._browser = _FakeBrowser(self._urls, self._bodies)
        return self._browser


class _FakePlaywright:
    def __init__(self, urls: list[str], bodies: list[str]) -> None:
        self._urls = urls
        self._bodies = bodies
        self._chromium = _FakeChromium(urls, bodies)

    def __enter__(self) -> _FakePlaywright:
        return self

    def __exit__(self, *args) -> None:
        return None

    @property
    def chromium(self) -> _FakeChromium:
        return self._chromium


def _fake_playwright_factory(urls: list[str], bodies: list[str]):
    def factory() -> _FakePlaywright:
        return _FakePlaywright(urls, bodies)

    return factory


class TestSavedLinkNormalization:
    def test_accepts_captured_analysis_url(self) -> None:
        assert _normalize_saved_link(
            "https://pagespeed.web.dev/analysis/https-example-com/b68uyhcpa1"
            "?form_factor=mobile"
        ) == (
            "https://pagespeed.web.dev/analysis/https-example-com/b68uyhcpa1"
            "?form_factor=mobile"
        )

    def test_strips_unknown_query(self) -> None:
        assert _normalize_saved_link(
            "https://pagespeed.web.dev/analysis/https-example-com/b68uyhcpa1?x=1"
        ) == "https://pagespeed.web.dev/analysis/https-example-com/b68uyhcpa1?form_factor=mobile"

    def test_rejects_non_analysis_urls(self) -> None:
        assert _normalize_saved_link("https://pagespeed.web.dev/") is None
        assert _normalize_saved_link(
            "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com"
        ) is None
        assert _normalize_saved_link("https://evil.example/x/y") is None
        assert _normalize_saved_link("") is None


class TestResolveSavedLink:
    def test_returns_saved_link_after_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start = "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        saved = (
            "https://pagespeed.web.dev/analysis/https-example-com/b68uyhcpa1"
            "?form_factor=mobile"
        )
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright",
            _fake_playwright_factory(
                [start, saved],
                ["PageSpeed Insights | Analyze", "Report from Aug 27, 2026"],
            ),
        )
        result = resolve_pagespeed_web_saved_link(
            "https://example.com/", timeout_seconds=0.5
        )
        assert result == saved

    def test_incomplete_report_yields_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start = "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        saved = (
            "https://pagespeed.web.dev/analysis/https-example-com/b68uyhcpa1"
            "?form_factor=mobile"
        )
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright",
            _fake_playwright_factory(
                [start, saved],
                ["PageSpeed Insights", "Analyzing..."],
            ),
        )
        result = resolve_pagespeed_web_saved_link(
            "https://example.com/", timeout_seconds=0.5
        )
        assert result is None

    def test_no_analysis_url_yields_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start = "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright",
            _fake_playwright_factory([start], ["PageSpeed Insights"]),
        )
        assert (
            resolve_pagespeed_web_saved_link(
                "https://example.com/", timeout_seconds=0.5
            )
            is None
        )

    def test_closes_browser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        start = "https://pagespeed.web.dev/analysis?url=https%3A%2F%2Fexample.com%2F"
        saved = (
            "https://pagespeed.web.dev/analysis/https-example-com/b68uyhcpa1"
            "?form_factor=mobile"
        )
        fake = _FakePlaywright(
            [start, saved],
            ["PageSpeed Insights", "Report from Aug 27, 2026"],
        )
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright", lambda: fake
        )
        resolve_pagespeed_web_saved_link("https://example.com/", timeout_seconds=0.5)
        browser = fake.chromium._browser
        assert browser.closed is True


class TestWebLinksCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cache = WebLinksCache(tmp_path)
        assert cache.load("https://example.com/") is None
        cache.store(
            "https://example.com/",
            "https://pagespeed.web.dev/analysis/https-example-com/x?form_factor=mobile",
        )
        assert cache.load("https://example.com/") == (
            "https://pagespeed.web.dev/analysis/https-example-com/x?form_factor=mobile"
        )
        assert cache.load("https://other.example/") is None

    def test_separate_urls_are_isolated(self, tmp_path: Path) -> None:
        cache = WebLinksCache(tmp_path)
        cache.store("https://a.example/", "https://pagespeed.web.dev/analysis/a/z")
        cache.store("https://b.example/", "https://pagespeed.web.dev/analysis/b/y")
        assert cache.load("https://a.example/") == "https://pagespeed.web.dev/analysis/a/z"
        assert cache.load("https://b.example/") == "https://pagespeed.web.dev/analysis/b/y"

    def test_store_entry_roundtrip(self, tmp_path: Path) -> None:
        cache = WebLinksCache(tmp_path)
        cache.store_entry(
            "https://example.com/",
            {
                "link": "https://pagespeed.web.dev/analysis/https-example-com/x?form_factor=mobile",
                "captured_at": "2026-08-30T12:00:00Z",
                "scores": {"mobile": 36, "desktop": 63},
            },
        )
        entry = cache.load_entry("https://example.com/")
        assert entry["link"] == (
            "https://pagespeed.web.dev/analysis/https-example-com/x?form_factor=mobile"
        )
        assert entry["captured_at"] == "2026-08-30T12:00:00Z"
        assert entry["scores"] == {"mobile": 36, "desktop": 63}

    def test_legacy_string_entry_is_migrated(self, tmp_path: Path) -> None:
        path = tmp_path / CACHE_DIRNAME / "web-links.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"https://example.com/": "https://pagespeed.web.dev/analysis/a/z"}),
            encoding="utf-8",
        )
        entry = WebLinksCache(tmp_path).load_entry("https://example.com/")
        assert entry == {
            "link": "https://pagespeed.web.dev/analysis/a/z",
            "captured_at": None,
            "scores": {},
        }

    def test_malformed_entries_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / CACHE_DIRNAME / "web-links.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "https://a.example/": {"scores": {"mobile": 36}},
                    "https://b.example/": 42,
                }
            ),
            encoding="utf-8",
        )
        cache = WebLinksCache(tmp_path)
        assert cache.load_entry("https://a.example/") is None
        assert cache.load_entry("https://b.example/") is None
        assert cache.load("https://c.example/") is None


class TestEnrichWebLinks:
    def test_cached_capture_is_reused_without_resolution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache = WebLinksCache(tmp_path)
        cache.store_entry(
            "https://example.com/",
            {
                "link": "https://pagespeed.web.dev/analysis/https-example-com/cached1?form_factor=mobile",
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scores": {"mobile": 36, "desktop": 63},
            },
        )
        called: list[str] = []
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_report",
            lambda url, timeout_seconds=240.0: called.append(url) or None,
        )
        report = enrich_pagespeed_web_links(
            {
                "schema_version": PAGESPEED_SCHEMA_VERSION,
                "urls": [{"url": "https://example.com/", "strategies": {}}],
            },
            cache_root=tmp_path,
        )
        entry = report["urls"][0]
        assert called == []
        assert entry["pagespeed_web_url"] == (
            "https://pagespeed.web.dev/analysis/https-example-com/cached1?form_factor=mobile"
        )
        assert entry["pagespeed_web_saved"] is True
        assert entry["pagespeed_web_fresh_url"] == pagespeed_web_url(
            "https://example.com/"
        )
        assert entry["saved_report_scores"] == {"mobile": 36, "desktop": 63}
        assert entry["saved_report_captured_at"] is not None

    def test_resolves_and_stores_when_uncached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        saved = "https://pagespeed.web.dev/analysis/https-example-com/fresh1?form_factor=mobile"
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_report",
            lambda url, timeout_seconds=240.0: {
                "link": saved,
                "captured_at": "2026-08-30T12:00:00Z",
                "scores": {"mobile": 36, "desktop": 63},
            },
        )
        report = enrich_pagespeed_web_links(
            {
                "schema_version": PAGESPEED_SCHEMA_VERSION,
                "urls": [{"url": "https://example.com/", "strategies": {}}],
            },
            cache_root=tmp_path,
        )
        entry = report["urls"][0]
        assert entry["pagespeed_web_url"] == saved
        assert entry["pagespeed_web_saved"] is True
        assert entry["saved_report_scores"] == {"mobile": 36, "desktop": 63}
        cached = WebLinksCache(tmp_path).load_entry("https://example.com/")
        assert cached["link"] == saved
        assert cached["scores"] == {"mobile": 36, "desktop": 63}

    def test_stale_capture_is_re_resolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache = WebLinksCache(tmp_path)
        cache.store_entry(
            "https://example.com/",
            {
                "link": "https://pagespeed.web.dev/analysis/https-example-com/stale1?form_factor=mobile",
                "captured_at": "2020-01-01T00:00:00Z",
                "scores": {"mobile": 99, "desktop": 99},
            },
        )
        saved = "https://pagespeed.web.dev/analysis/https-example-com/fresh2?form_factor=mobile"
        called: list[str] = []
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_report",
            lambda url, timeout_seconds=240.0: called.append(url)
            or {
                "link": saved,
                "captured_at": "2026-08-30T12:00:00Z",
                "scores": {"mobile": 40, "desktop": 63},
            },
        )
        report = enrich_pagespeed_web_links(
            {
                "schema_version": PAGESPEED_SCHEMA_VERSION,
                "urls": [{"url": "https://example.com/", "strategies": {}}],
            },
            cache_root=tmp_path,
        )
        entry = report["urls"][0]
        assert called == ["https://example.com/"]
        assert entry["pagespeed_web_url"] == saved
        assert entry["saved_report_scores"] == {"mobile": 40, "desktop": 63}

    def test_resolve_disabled_keeps_legacy_cached_link(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = tmp_path / CACHE_DIRNAME / "web-links.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"https://example.com/": "https://pagespeed.web.dev/analysis/a/z"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_report",
            lambda url, timeout_seconds=240.0: pytest.fail("must not resolve"),
        )
        report = enrich_pagespeed_web_links(
            {
                "schema_version": PAGESPEED_SCHEMA_VERSION,
                "urls": [{"url": "https://example.com/", "strategies": {}}],
            },
            cache_root=tmp_path,
            resolve=False,
        )
        entry = report["urls"][0]
        assert entry["pagespeed_web_url"] == "https://pagespeed.web.dev/analysis/a/z"
        assert entry["pagespeed_web_saved"] is True
        assert entry["saved_report_scores"] == {}

    def test_failure_falls_back_to_fresh_link(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def failing(url, timeout_seconds=240.0) -> None:
            raise RuntimeError("browser unavailable")

        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_report", failing
        )
        report = enrich_pagespeed_web_links(
            {
                "schema_version": PAGESPEED_SCHEMA_VERSION,
                "urls": [{"url": "https://example.com/", "strategies": {}}],
            },
            cache_root=tmp_path,
        )
        entry = report["urls"][0]
        assert entry["pagespeed_web_url"] == pagespeed_web_url("https://example.com/")
        assert entry["pagespeed_web_saved"] is False
        assert entry["saved_report_scores"] == {}

    def test_resolve_disabled_uses_fresh_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_report",
            lambda url, timeout_seconds=240.0: called.append(url) or "x",
        )
        report = enrich_pagespeed_web_links(
            {
                "schema_version": PAGESPEED_SCHEMA_VERSION,
                "urls": [{"url": "https://example.com/", "strategies": {}}],
            },
            cache_root=tmp_path,
            resolve=False,
        )
        assert called == []
        assert report["urls"][0]["pagespeed_web_saved"] is False


class TestSavedReportCapture:
    def test_resolve_report_combines_link_and_scores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saved = "https://pagespeed.web.dev/analysis/https-example-com/abc?form_factor=mobile"
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_link",
            lambda url, timeout_seconds=240.0: saved,
        )
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.extract_pagespeed_web_saved_scores",
            lambda link: {"mobile": 36, "desktop": 63},
        )
        captured = resolve_pagespeed_web_saved_report("https://example.com/")
        assert captured is not None
        assert captured["link"] == saved
        assert captured["scores"] == {"mobile": 36, "desktop": 63}
        assert captured["captured_at"]

    def test_score_extraction_failure_keeps_link(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saved = "https://pagespeed.web.dev/analysis/https-example-com/abc?form_factor=mobile"
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_link",
            lambda url, timeout_seconds=240.0: saved,
        )

        def failing(link: str) -> dict[str, int | None]:
            raise RuntimeError("gauge never rendered")

        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.extract_pagespeed_web_saved_scores", failing
        )
        captured = resolve_pagespeed_web_saved_report("https://example.com/")
        assert captured is not None
        assert captured["link"] == saved
        assert captured["scores"] == {}

    def test_unresolved_link_yields_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.resolve_pagespeed_web_saved_link",
            lambda url, timeout_seconds=240.0: None,
        )
        assert resolve_pagespeed_web_saved_report("https://example.com/") is None


class TestSavedScoreExtraction:
    @staticmethod
    def _articles_playwright(
        payloads: list[dict], *, fail: bool = False
    ) -> object:
        class _ArticlesPage:
            def __init__(self) -> None:
                self._index = 0

            def goto(self, *args, **kwargs) -> None:
                pass

            def click(self, *args, **kwargs) -> None:
                raise RuntimeError("no consent dialog")

            def wait_for_timeout(self, milliseconds: int) -> None:
                self._index += 1

            def evaluate(self, script: str) -> dict:
                if fail:
                    raise RuntimeError("page unavailable")
                return payloads[min(self._index, len(payloads) - 1)]

        class _Context:
            def new_page(self) -> _ArticlesPage:
                return _ArticlesPage()

        class _Browser:
            closed = False

            def new_context(self) -> _Context:
                return _Context()

            def close(self) -> None:
                self.closed = True

        class _Chromium:
            def launch(self) -> _Browser:
                return _Browser()

        class _Playwright:
            chromium = _Chromium()

            def __enter__(self) -> _Playwright:
                return self

            def __exit__(self, *args) -> None:
                return None

        return _Playwright()

    def test_extracts_scores_per_strategy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saved = "https://pagespeed.web.dev/analysis/https-example-com/abc?form_factor=mobile"
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright",
            lambda: self._articles_playwright(
                [
                    {"mobile": 36, "desktop": 63},
                ]
            ),
        )
        scores = extract_pagespeed_web_saved_scores(saved)
        assert scores == {"mobile": 36, "desktop": 63}

    def test_out_of_range_scores_are_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saved = "https://pagespeed.web.dev/analysis/https-example-com/abc?form_factor=mobile"
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright",
            lambda: self._articles_playwright(
                [
                    {"mobile": 255, "desktop": -1},
                ]
            ),
        )
        # The fake page never renders valid gauges, so the poll loop would
        # otherwise run its full 120s production deadline against a no-op
        # fake; one poll is enough to pin the filtering behavior.
        scores = extract_pagespeed_web_saved_scores(saved, timeout_seconds=1)
        assert scores == {}

    def test_page_failure_yields_no_scores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saved = "https://pagespeed.web.dev/analysis/https-example-com/abc?form_factor=mobile"
        monkeypatch.setattr(
            "playwright.sync_api.sync_playwright",
            lambda: self._articles_playwright([], fail=True),
        )
        scores = extract_pagespeed_web_saved_scores(saved)
        assert scores == {}


class TestBuildUrlReport:
    async def test_ok_strategy_assembles_bounded_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_fetch(url, strategy, **kwargs):
            return _lighthouse(
                {
                    "failed-audit": _audit("failed-audit", mode="numeric", score=0.2),
                    "passed-audit": _audit("passed-audit", mode="binary", score=1.0),
                },
                categories={
                    "performance": {
                        "id": "performance",
                        "title": "Performance",
                        "score": 0.42,
                        "displayValue": "42 s",
                        "auditRefs": [{}] * 47,
                    }
                },
            )

        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.fetch_pagespeed", fake_fetch
        )
        report = await build_url_report("https://example.com", ("mobile", "desktop"))
        assert report["url"] == "https://example.com"
        assert report["ok_strategies"] == ["mobile", "desktop"]
        assert report["pagespeed_web_url"] == pagespeed_web_url(
            "https://example.com"
        )
        mobile = report["strategies"]["mobile"]
        assert mobile["status"] == "ok"
        assert mobile["from_cache"] is False
        assert mobile["lighthouse_version"] == "12.1.0"
        assert mobile["categories"][0]["score_percent"] == 42
        assert mobile["audits"]["totals"] == {
            "failed": 1,
            "passed": 1,
            "not_applicable": 0,
            "manual": 0,
            "informative": 0,
            "error": 0,
        }
        assert "audits" in mobile and "opportunities" in mobile

    async def test_failed_strategy_becomes_error_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_fetch(url, strategy, **kwargs):
            raise PagespeedApiError("API request failed (HTTP 500)", status_code=500)

        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.fetch_pagespeed", fake_fetch
        )
        report = await build_url_report("https://example.com", ("mobile",))
        entry = report["strategies"]["mobile"]
        assert entry["status"] == "error"
        assert "HTTP 500" in entry["error"]
        assert report["ok_strategies"] == []

    async def test_cache_hit_marks_from_cache_and_skips_fetch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache = PagespeedCache(tmp_path)
        cache.store(
            "https://example.com",
            "mobile",
            _lighthouse({"cached": _audit("cached")}),
        )
        calls: list[str] = []

        async def fake_fetch(url, strategy, **kwargs):
            calls.append(strategy)
            return _lighthouse({})

        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.fetch_pagespeed", fake_fetch
        )
        report = await build_url_report(
            "https://example.com", ("mobile",), cache=cache
        )
        assert report["strategies"]["mobile"]["from_cache"] is True
        assert calls == []


class TestPagespeedReport:
    async def test_deduplicates_urls_and_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        async def fake_fetch(url, strategy, **kwargs):
            seen.append(url)
            raise PagespeedApiError("API request failed (HTTP 429)", status_code=429)

        monkeypatch.setattr(
            "ux_analyzer.analysis.pagespeed.fetch_pagespeed", fake_fetch
        )
        report = await pagespeed_report(
            ("https://a.example/", "https://a.example/", "", "https://b.example/"),
            strategies=("mobile",),
        )
        assert report["schema_version"] == PAGESPEED_SCHEMA_VERSION
        assert report["url_count"] == 2
        assert report["ok_strategy_count"] == 0
        assert seen == ["https://a.example/", "https://b.example/"]

    def test_sync_wrapper(self, tmp_path: Path) -> None:
        cache = PagespeedCache(tmp_path)
        cache.store("https://a.example/", "mobile", _lighthouse({}))
        report = pagespeed_report_sync(
            ("https://a.example/",), cache_root=tmp_path, strategies=("mobile",)
        )
        assert report["ok_strategy_count"] == 1
        assert report["urls"][0]["strategies"]["mobile"]["status"] == "ok"
        assert report["urls"][0]["strategies"]["mobile"]["from_cache"] is True

