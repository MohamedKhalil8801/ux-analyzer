"""Live-page UX audit across application start URLs for experiment reports."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ux_analyzer.analysis.accessibility import analyze_accessibility
from ux_analyzer.analysis.geo import analyze_geo
from ux_analyzer.analysis.imagery import analyze_imagery
from ux_analyzer.analysis.meta_semantic import analyze_meta_semantic
from ux_analyzer.analysis.performance import analyze_performance

AUDIT_SCHEMA_VERSION = "ux-audit-v1"
AUDIT_FILENAME = "ux-audit.json"

_CATEGORIES = (
    ("GEO", analyze_geo),
    ("meta-semantic", analyze_meta_semantic),
    ("performance", analyze_performance),
    ("accessibility", analyze_accessibility),
    ("imagery", analyze_imagery),
)


async def audit_url(url: str) -> dict[str, Any]:
    """Run every expanded detector against one URL and return its report."""

    results = await asyncio.gather(
        *(analyze(url) for _, analyze in _CATEGORIES)
    )
    issues: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for (category, _), category_issues in zip(_CATEGORIES, results, strict=True):
        counts[category] = len(category_issues)
        for issue in category_issues:
            issues.append(
                {
                    "category": category,
                    "check_id": issue.check_id,
                    "title": issue.title,
                    "severity": issue.severity,
                    "evidence": issue.evidence,
                }
            )
    return {
        "url": url,
        "counts": counts,
        "total": len(issues),
        "issues": issues,
    }


async def audit_urls(urls: Sequence[str]) -> dict[str, Any]:
    """Audit each unique URL; per-URL failures stay bounded and explicit."""

    unique_urls = tuple(dict.fromkeys(url for url in urls if url))
    url_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for url in unique_urls:
        try:
            url_reports.append(await audit_url(url))
        except Exception as error:  # noqa: BLE001 - audit never fails a run
            errors.append(
                {
                    "url": url,
                    "error": f"{type(error).__name__}: {error}"[:512],
                }
            )
    total = sum(report["total"] for report in url_reports)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "total_issues": total,
        "urls": url_reports,
        "errors": errors,
    }


def audit_urls_sync(urls: Sequence[str]) -> dict[str, Any]:
    """Synchronous wrapper for report-time auditing."""

    return asyncio.run(audit_urls(urls))
