"""Score ux-analyzer visual detectors against the UEye answer key.

UEye scoring: selecting a misapplied fundamental earns +1; selecting a
correctly-applied one loses 1 point. This harness mirrors that and also
reports precision/recall/F1 per fundamental and overall.

Usage:
    python benchmarks/ueye/evaluate.py [--snapshots benchmarks/ueye/evidence]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from ux_analyzer.analysis.visual.pipeline import (  # noqa: E402
    analyze_snapshot,
    fundamentals_flagged,
)
from ux_analyzer.analysis.visual.snapshot import load_snapshot_json  # noqa: E402

FUNDAMENTALS = (
    "white-space",
    "contrast",
    "color",
    "typography",
    "scale",
    "alignment",
    "visual-hierarchy",
)


def evaluate(base: pathlib.Path) -> dict:
    answers = json.loads((base / "answers.json").read_text(encoding="utf-8"))
    per_fixture = {}
    tp = fp = fn = tn = 0
    points = 0
    per_fund: dict[str, dict[str, int]] = {
        f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in FUNDAMENTALS
    }
    issue_log: dict[str, list[dict]] = {}
    for slug, key in sorted(answers.items()):
        snap_path = base / "evidence" / f"{slug}.styles.json"
        snapshot = load_snapshot_json(snap_path)
        issues = [i for i in analyze_snapshot(snapshot)]
        flagged = fundamentals_flagged(issues)
        expected = {f for f in FUNDAMENTALS if key.get(f)}
        hit = flagged & expected
        miss = expected - flagged
        alarm = flagged - expected
        quiet = set(FUNDAMENTALS) - expected - flagged
        pts = len(hit) + sum(-1 for _ in miss) + sum(-1 for _ in alarm) + (1 if not expected and not flagged else 0)
        # clean-design bonus: nothing-wrong true means expected == empty
        tp += len(hit)
        fn += len(miss)
        fp += len(alarm)
        tn += len(quiet)
        points += pts
        for f in FUNDAMENTALS:
            if f in hit:
                per_fund[f]["tp"] += 1
            if f in miss:
                per_fund[f]["fn"] += 1
            if f in alarm:
                per_fund[f]["fp"] += 1
            if f in quiet:
                per_fund[f]["tn"] += 1
        per_fixture[slug] = {
            "expected": sorted(expected),
            "flagged": sorted(flagged),
            "hit": sorted(hit),
            "missed": sorted(miss),
            "false_alarms": sorted(alarm),
            "points": pts,
            "issues": [
                {
                    "fundamental": i.fundamental,
                    "check_id": i.check_id,
                    "title": i.title,
                    "severity": i.severity,
                    "elements": list(i.element_refs)[:6],
                    "evidence": i.evidence,
                }
                for i in issues
            ],
        }
        issue_log[slug] = per_fixture[slug]["issues"]

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "ueye_points": points,
        "max_points": 28,
        "micro": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)},
        "per_fundamental": per_fund,
        "per_fixture": per_fixture,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# UEye evaluation",
        "",
        f"**UEye-style points: {report['ueye_points']} / {report['max_points']}**",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    m = report["micro"]
    lines += [
        f"| precision | {m['precision']:.3f} |",
        f"| recall | {m['recall']:.3f} |",
        f"| f1 | {m['f1']:.3f} |",
        "",
        "| fundamental | tp | fp | fn | tn |",
        "|---|---|---|---|---|",
    ]
    for f, v in report["per_fundamental"].items():
        lines.append(f"| {f} | {v['tp']} | {v['fp']} | {v['fn']} | {v['tn']} |")
    lines += ["", "| fixture | missed | false alarms | pts |", "|---|---|---|---|"]
    for slug, r in report["per_fixture"].items():
        lines.append(
            f"| {slug} | {','.join(r['missed']) or '—'} | {','.join(r['false_alarms']) or '—'} | {r['points']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="benchmarks/ueye")
    args = ap.parse_args()
    base = pathlib.Path(args.base)
    report = evaluate(base)
    out = base / "reports"
    out.mkdir(exist_ok=True)
    (out / "eval.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = render_markdown(report)
    (out / "eval.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
