"""Blind A/B comparison of oracle (npx slop-detect) vs ours (uxa slop).

Usage:
  python scripts/slop_bench/compare.py [--slugs a,b,c] [--blind]
    --blind  relabels the two sides A/B with identities stripped, writes
             scripts/slop_bench/compare/blind_AB.md for the critic.

Outputs per-slug: score/tier/unified, triggered pattern diff, per-pattern
trigger parity (TP/TN/FP/FN vs oracle), and copy-axis diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from corpus import CORPUS, ORACLE_DIR, OURS_DIR  # noqa: E402


def load(slug: str, which: str) -> dict:
    path = ROOT / (ORACLE_DIR if which == "oracle" else OURS_DIR) / f"{slug}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def dirs(frozen: bool) -> tuple[str, str]:
    return ("oracle_frozen", "ours_frozen") if frozen else (ORACLE_DIR, OURS_DIR)


def pattern_map(report: dict) -> dict[str, bool]:
    return {p["id"]: bool(p["triggered"]) for p in report.get("patterns", [])}


def copy_map(report: dict) -> dict[str, bool]:
    copy = report.get("copy") or {}
    if not copy and report.get("axes") and report["axes"].get("copy"):
        copy = report["axes"]["copy"]
    return {p["id"]: bool(p["triggered"]) for p in copy.get("patterns", [])}


def compare_slug(slug: str, odir: str, mdir: str) -> dict:
    oracle = load(slug, "oracle") if odir == ORACLE_DIR else json.loads((ROOT / odir / f"{slug}.json").read_text(encoding="utf-8"))
    ours = load(slug, "ours") if mdir == OURS_DIR else json.loads((ROOT / mdir / f"{slug}.json").read_text(encoding="utf-8"))
    o_map = pattern_map(oracle)
    m_map = pattern_map(ours)
    ids = sorted(set(o_map) | set(m_map))
    triggered = {"oracle": [i for i in ids if o_map.get(i)], "ours": [i for i in ids if m_map.get(i)]}
    tp = [i for i in ids if o_map.get(i) and m_map.get(i)]
    fn = [i for i in ids if o_map.get(i) and not m_map.get(i)]
    fp = [i for i in ids if not o_map.get(i) and m_map.get(i)]
    tn = [i for i in ids if not o_map.get(i) and not m_map.get(i)]
    o_copy = copy_map(oracle)
    m_copy = copy_map(ours)
    cids = sorted(set(o_copy) | set(m_copy))
    copy_tp = [i for i in cids if o_copy.get(i) and m_copy.get(i)]
    copy_fn = [i for i in cids if o_copy.get(i) and not m_copy.get(i)]
    copy_fp = [i for i in cids if not o_copy.get(i) and m_copy.get(i)]
    copy_tn = [i for i in cids if not o_copy.get(i) and not m_copy.get(i)]
    oracle_copy = oracle.get("copy") or (oracle.get("axes") or {}).get("copy") or {}
    ours_copy = ours.get("copy") or {}
    unified = oracle.get("unifiedScore")
    return {
        "slug": slug,
        "oracle": {
            "score": oracle.get("score"),
            "tier": oracle.get("tier"),
            "grade": oracle.get("grade"),
            "unified": unified,
            "copy": {"score": oracle_copy.get("score"), "tier": oracle_copy.get("tier")},
        },
        "ours": {
            "score": ours.get("score"),
            "tier": ours.get("tier"),
            "grade": ours.get("grade"),
            "unified": ours.get("unifiedScore"),
            "copy": {"score": ours_copy.get("score"), "tier": ours_copy.get("tier")},
        },
        "design": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "copy": {"tp": copy_tp, "fn": copy_fn, "fp": copy_fp, "tn": copy_tn},
        "triggered_oracle": triggered["oracle"],
        "triggered_ours": triggered["ours"],
        "matches_oracle": len(tp),
        "totals": {"oracle": len(triggered["oracle"]), "ours": len(triggered["ours"])},
    }


def summarize(rows: list[dict]) -> dict:
    design = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    copy = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tiers_match = 0
    for r in rows:
        for k in design:
            design[k] += len(r["design"][k])
        for k in copy:
            copy[k] += len(r["copy"][k])
        if r["oracle"]["tier"] == r["ours"]["tier"]:
            tiers_match += 1
    return {
        "sites": len(rows),
        "design": design,
        "copy": copy,
        "tiers_match": tiers_match,
        "precision": round(design["tp"] / max(design["tp"] + design["fp"], 1), 3),
        "recall": round(design["tp"] / max(design["tp"] + design["fn"], 1), 3),
    }


def fmt_side(
    label: str,
    r: dict,
    key: str,
    pattern_ids: set[str] | None = None,
    odir: str = ORACLE_DIR,
    mdir: str = OURS_DIR,
) -> list[str]:
    lines = []
    meta = r[key]
    lines.append(f"### Detector {label}")
    lines.append(
        f"- design: **{meta['tier']}** score {meta['score']}/100 · {meta['grade']} · "
        f"{len(r['triggered_oracle'] if key == 'oracle' else r['triggered_ours'])}/{27} patterns"
    )
    unif = meta.get("unified")
    if unif is not None:
        lines.append(f"- unified (design+copy): {unif}/100")
    copy = meta.get("copy")
    if copy and copy.get("score") is not None:
        lines.append(f"- copy: {copy['tier']} score {copy['score']}/100")
    triggered = r["triggered_oracle"] if key == "oracle" else r["triggered_ours"]
    if pattern_ids:
        triggered = [i for i in triggered if i in pattern_ids]
    lines.append("- triggered: " + (", ".join(triggered) if triggered else "(none)"))
    evidence = load_evidence(r["slug"], key, triggered, odir, mdir)
    for pid, ev in evidence:
        compact = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))[:220]
        lines.append(f"  - {pid}: {compact}")
    return lines


def load_evidence(slug: str, which: str, ids: list[str], odir: str, mdir: str) -> list[tuple[str, dict]]:
    """Fetch evidence objects for triggered patterns (best-effort)."""
    if which == "oracle":
        report = json.loads((ROOT / odir / f"{slug}.json").read_text(encoding="utf-8"))
        by_id = {p["id"]: p.get("evidence", {}) for p in report.get("patterns", [])}
    else:
        report = json.loads((ROOT / mdir / f"{slug}.json").read_text(encoding="utf-8"))
        by_id = {p["id"]: p.get("evidence", {}) for p in report.get("patterns", [])}
    out = []
    for pid in ids:
        ev = by_id.get(pid)
        if ev is not None:
            out.append((pid, ev))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", default="")
    ap.add_argument("--patterns", default="", help="comma-separated pattern ids to focus on")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument(
        "--frozen",
        action="store_true",
        help="compare frozen-DOM captures (oracle_frozen/ours_frozen)",
    )
    args = ap.parse_args()
    slugs = [s.strip() for s in args.slugs.split(",") if s.strip()] or [
        c["slug"] for c in CORPUS
    ]
    odir, mdir = dirs(args.frozen)
    focus = {p.strip() for p in args.patterns.split(",") if p.strip()} or None
    rows = [compare_slug(slug, odir, mdir) for slug in slugs]
    summary = summarize(rows)
    out = ROOT / "compare"
    out.mkdir(parents=True, exist_ok=True)

    if args.blind:
        doc = []
        for r in rows:
            doc.append(f"## Site {r['slug']}")
            doc.append(f"Expected lean: {next(c['lean'] for c in CORPUS if c['slug'] == r['slug'])}")
            doc.extend(fmt_side("A", r, "oracle", focus, odir, mdir))
            doc.extend(fmt_side("B", r, "ours", focus, odir, mdir))
            doc.append("")
        suffix = "_frozen" if args.frozen else ""
        piece = f"_{args.patterns.replace(',', '_')}" if args.patterns else ""
        target = out / f"blind_AB{suffix}{piece}.md"
        target.write_text("\n".join(doc), encoding="utf-8")
        print(f"blind A/B written: {target}")
        return

    for r in rows:
        d = r["design"]
        tier_mark = "OK" if r["oracle"]["tier"] == r["ours"]["tier"] else "DIFF"
        print(
            f"{r['slug']:<16} oracle {str(r['oracle']['score']):>3}/{r['oracle']['tier']:<6} "
            f"ours {str(r['ours']['score']):>3}/{r['ours']['tier']:<6} "
            f"tier{tier_mark} TP={len(d['tp'])} FP={len(d['fp'])} FN={len(d['fn'])}"
        )
        if d["fp"]:
            print(f"    FP: {', '.join(d['fp'])}")
        if d["fn"]:
            print(f"    FN: {', '.join(d['fn'])}")
    print("─" * 60)
    print(
        f"SUMMARY: design TP={summary['design']['tp']} FP={summary['design']['fp']} "
        f"FN={summary['design']['fn']} TN={summary['design']['tn']} "
        f"precision={summary['precision']} recall={summary['recall']} "
        f"tiers_match={summary['tiers_match']}/{summary['sites']}"
    )
    print(
        f"COPY: TP={summary['copy']['tp']} FP={summary['copy']['fp']} FN={summary['copy']['fn']}"
    )
    (out / "summary.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
