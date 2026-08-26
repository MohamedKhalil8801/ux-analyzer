"""Build anonymized A/B packets for blind critic judging.

For each judged piece, produce a markdown packet where:
- side A and side B are the detector output and the UEye expected issues
  (which is which is randomized per run)
- all source-identifying labels are stripped
The critic reads the packet plus the fixture screenshots and picks the more
accurate side, then names the single biggest gap.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

PIECES: dict[str, tuple[str, ...]] = {
    "spacing-layout": ("white-space",),
    "typography-scale": ("typography", "scale"),
    "color-contrast": ("contrast", "color"),
    "hierarchy-alignment": ("alignment", "visual-hierarchy"),
}

_CANONICAL = {
    "white-space": "White space is misapplied in this design (cramped, inconsistent, or missing grouping).",
    "contrast": "Contrast is misapplied in this design (text or elements lack sufficient contrast).",
    "color": "Color is misapplied in this design (palette/accent usage is off).",
    "typography": "Typography is misapplied in this design (font choices/usage are off).",
    "scale": "Scale is misapplied in this design (element sizes are off relative to each other).",
    "alignment": "Alignment is misapplied in this design (edges or baselines do not line up).",
    "visual-hierarchy": "Visual hierarchy is misapplied in this design (importance is not communicated).",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="benchmarks/ueye")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    base = pathlib.Path(args.base)
    report = json.loads((base / "reports" / "eval.json").read_text(encoding="utf-8"))
    answers = json.loads((base / "answers.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    out_dir = base / "reports" / "blind"
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = {}
    for piece, funds in PIECES.items():
        lines = [f"# Blind judgment packet — {piece}", ""]
        for slug, data in report["per_fixture"].items():
            ours = [
                f"- [{i['severity']}] {i['title']}"
                for i in data["issues"]
                if i["fundamental"] in funds
            ]
            theirs = [
                f"- {_CANONICAL[f]}"
                for f in funds
                if answers.get(slug, {}).get(f) and slug in answers
            ]
            if not ours and not theirs:
                continue
            a, b = ("OURS", "UEYE") if rng.random() < 0.5 else ("UEYE", "OURS")
            sides = {"A": [], "B": []}
            sides["A" if a == "OURS" else "B"] = ours
            sides["A" if b == "OURS" else "B"] = theirs
            lines += [
                f"## Fixture: {slug}  (screenshot: evidence/{slug}.png)",
                "",
                "### Side A",
                *(sides["A"] or ["- No issues reported."]),
                "",
                "### Side B",
                *(sides["B"] or ["- No issues reported."]),
                "",
            ]
            truth[slug] = {"A": a, "B": b}
        (out_dir / f"{piece}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {piece}.md")
    (out_dir / "_truth.json").write_text(
        json.dumps(truth, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
