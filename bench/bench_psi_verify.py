"""Blind A/B: derive our report from the same oracle JSON the critic holds.

Usage:
    python bench_psi_verify.py <oracle.json> [more...]

For each oracle file: source = raw PSI API JSON; derived = produced by our
extraction pipeline FROM THE SAME FILE. Every value in derived must equal a
value in source (no invented numbers). Prints a verdict per check family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ux_analyzer.analysis.pagespeed import (  # noqa: E402
    PSI_PASS_THRESHOLD,
    aggregate_categories,
    extract_audits,
    extract_field_data,
    extract_opportunities,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def num_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 1e-9


def verify(oracle_path: Path) -> None:
    print(f"== {oracle_path.name} ==")
    raw = json.loads(oracle_path.read_text(encoding="utf-8"))
    lr = raw.get("lighthouseResult") or {}

    # --- categories ---
    derived_cats = {c["id"]: c for c in aggregate_categories(raw)}
    source_cats = lr.get("categories") or {}
    for cid, scat in source_cats.items():
        dcat = derived_cats.get(cid)
        check(
            f"category {cid} present",
            dcat is not None,
            f"missing in derived (source had {sorted(derived_cats)})",
        )
        if dcat is None:
            continue
        check(f"category {cid} score", num_equal(dcat["score"], scat.get("score")))
        check(
            f"category {cid} score_percent",
            dcat["score_percent"] is not None
            and scat.get("score") is not None
            and dcat["score_percent"] == round(float(scat["score"]) * 100),
        )
        check(
            f"category {cid} title",
            dcat["title"] == scat.get("title"),
            f"{dcat['title']!r} != {scat.get('title')!r}",
        )
        check(
            f"category {cid} display_value",
            dcat["display_value"] == scat.get("displayValue"),
        )
        check(
            f"category {cid} audit_refs count",
            dcat["audit_refs"] == len(scat.get("auditRefs") or []),
        )
    check(
        "no invented categories",
        set(derived_cats) <= set(source_cats),
        f"extra: {sorted(set(derived_cats) - set(source_cats))}",
    )

    # --- audits ---
    buckets = extract_audits(raw)
    flat = {a["id"]: (bucket, a) for bucket, rows in buckets.items() for a in rows}
    source_audits = lr.get("audits") or {}
    scored_modes = {"binary", "numeric", "metricSavings", "passableAudit"}
    total_bucket_mismatch = 0
    for aid, sa in source_audits.items():
        mode = sa.get("scoreDisplayMode") or "numeric"
        score = sa.get("score")
        if mode in scored_modes:
            expected = "error" if score is None else (
                "failed" if score < PSI_PASS_THRESHOLD else "passed"
            )
        elif mode == "notApplicable":
            expected = "not_applicable"
        elif mode == "manual":
            expected = "manual"
        elif mode == "informative":
            expected = "informative"
        elif mode == "error":
            expected = "error"
        else:
            expected = "numeric"
        got_bucket, row = flat.get(aid, (None, None))
        if got_bucket != expected:
            total_bucket_mismatch += 1
            check(
                f"audit {aid} bucket",
                False,
                f"expected {expected}, got {got_bucket} (mode={mode}, score={score})",
            )
        elif row is not None:
            check(f"audit {aid} score", num_equal(row["score"], score))
            check(f"audit {aid} title", row["title"] == sa.get("title"))
            check(
                f"audit {aid} display_value",
                row["display_value"] == sa.get("displayValue"),
            )
            check(
                f"audit {aid} description",
                row["description"] == sa.get("description"),
            )
    check(
        "no invented audits",
        set(flat) <= set(source_audits),
        f"extra: {sorted(set(flat) - set(source_audits))[:10]}",
    )
    total_buckets = len(source_audits)
    check(
        "all source audits classified",
        len(flat) == total_buckets,
        f"{len(flat)} derived vs {total_buckets} source",
    )

    # --- opportunities / savings ---
    opps = extract_opportunities(raw)
    all_opp = {o["id"]: o for o in opps["opportunities"]}
    source_opps = 0
    for aid, sa in source_audits.items():
        details = sa.get("details")
        if not isinstance(details, dict) or details.get("type") != "opportunity":
            continue
        source_opps += 1
        row = all_opp.get(aid)
        check(f"opportunity {aid} present", row is not None)
        if row is None:
            continue
        check(
            f"opportunity {aid} savings_ms",
            num_equal(row["savings_ms"], details.get("overallSavingsMs")),
            f"derived {row['savings_ms']} vs source {details.get('overallSavingsMs')}",
        )
        check(
            f"opportunity {aid} savings_bytes",
            num_equal(row["savings_bytes"], details.get("overallSavingsBytes")),
        )
        source_items = [
            {k: v for k, v in item.items() if k in ("url", "totalBytes", "wastedBytes", "wastedMs", "responseTime", "transferSize", "requestCount") and (isinstance(v, (int, float)) and not isinstance(v, bool) or isinstance(v, str) and k == "url")}
            for item in (details.get("items") or [])
            if isinstance(item, dict)
        ][:50]
        check(
            f"opportunity {aid} items",
            row["items"] == source_items,
            f"derived {row['items'][:2]} vs source {source_items[:2]}",
        )
    check(
        "no invented opportunities",
        set(all_opp) <= set(source_audits),
        f"extra: {sorted(set(all_opp) - set(source_audits))[:10]}",
    )
    ms = {o["id"]: o for o in opps["metric_savings"]}
    for aid, sa in source_audits.items():
        details = sa.get("details")
        if not isinstance(details, dict) or details.get("type") == "opportunity":
            continue
        has_savings = "overallSavingsMs" in details or "overallSavingsBytes" in details
        if not has_savings:
            continue
        row = ms.get(aid)
        check(f"metric_savings {aid} present", row is not None)
        if row is not None:
            check(
                f"metric_savings {aid} savings_ms",
                num_equal(row["savings_ms"], details.get("overallSavingsMs")),
            )
            check(
                f"metric_savings {aid} savings_bytes",
                num_equal(row["savings_bytes"], details.get("overallSavingsBytes")),
            )
    check(
        "no invented metric_savings",
        set(ms) <= set(source_audits),
        f"extra: {sorted(set(ms) - set(source_audits))[:10]}",
    )

    # --- field data ---
    fd = extract_field_data(raw)
    le = raw.get("loadingExperience")
    if isinstance(le, dict) and (le.get("overall_category") or le.get("metrics")):
        check("field data present", fd is not None)
        if fd is not None:
            check(
                "field overall_category",
                fd.get("overall_category") == le.get("overall_category"),
            )
            source_metrics = {
                mid: {k: m.get(k) for k in ("percentile", "category") if k in m}
                for mid, m in (le.get("metrics") or {}).items()
                if isinstance(m, dict)
            }
            check("field metrics", fd.get("metrics") == (source_metrics or None), f"{fd.get('metrics')} vs {source_metrics}")
    else:
        check("field data absent when source absent", fd is None)

    print()


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: bench_psi_verify.py <oracle.json> ...")
        return 2
    for path in paths:
        verify(path)
    if FAILURES:
        print(f"VERDICT: FAIL — {len(FAILURES)} mismatch(es)")
        return 1
    print("VERDICT: PASS — derived matches source on every checked value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
