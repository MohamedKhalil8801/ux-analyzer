"""Same-session A/B: oracle signals vs our detector on the identical DOM.

Usage: python scripts/slop_bench/same_session.py <slug-or-url>
Runs ref_runner/same_session.mjs (one page load capturing BOTH the published
detector's signals and our snapshot), then runs our Python detector on that
snapshot and diffs the two pattern-by-pattern.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from corpus import CORPUS  # noqa: E402

REF_RUNNER = ROOT / "ref_runner" / "same_session.mjs"
UXA_SRC = ROOT.parents[1] / "src"


def resolve(arg: str) -> str:
    for c in CORPUS:
        if c["slug"] == arg:
            return c["url"]
    return arg


def main() -> None:
    url = resolve(sys.argv[1])
    out_path = ROOT / "compare" / f"same_session_{Path(url).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["node", str(REF_RUNNER), url, str(out_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REF_RUNNER.parent),
    )
    if proc.returncode != 0 or not out_path.is_file():
        print(f"capture failed: {proc.stderr[-400:] or proc.stdout[-400:]}")
        sys.exit(1)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    oracle = data["oracle"]
    snapshot = data["snapshot"]
    meta = data["meta"]

    sys.path.insert(0, str(UXA_SRC))
    from ux_analyzer.analysis.slop.pipeline import analyze_slop
    from ux_analyzer.analysis.visual.snapshot import snapshot_from_dict

    ours = analyze_slop(
        snapshot_from_dict(snapshot),
        viewport_w=int(meta["viewport"]["w"]),
        viewport_h=int(meta["viewport"]["h"]),
        doc_height=int(meta["docHeight"] or 0),
        scroll_y=int(meta["scrollY"] or 0),
        text_context=meta["textContext"],
        surface=meta["surface"],
    )
    o_sig = oracle["signals"]
    diffs = []
    for p in ours["patterns"]:
        pid = p["id"]
        o_trig = bool(o_sig.get(pid, {}).get("triggered"))
        m_trig = p["triggered"]
        if o_trig != m_trig:
            diffs.append(
                (pid, o_trig, o_sig.get(pid), m_trig, p["evidence"])
            )
    o_flags = sorted(pid for pid, s in o_sig.items() if s.get("triggered"))
    m_flags = sorted(p["id"] for p in ours["patterns"] if p["triggered"])
    print(f"url: {url}")
    print(f"oracle flagged: {o_flags}")
    print(f"ours   flagged: {m_flags}")
    print(f"oracle score: {sum(next((p['weight'] for p in PATTERNS if p['id'] == pid), 0) for pid in o_flags)}")
    print(f"ours   score: {ours['score']} tier={ours['tier']}")
    if not diffs:
        print("PARITY: identical pattern verdicts on the same DOM")
    else:
        for pid, o_t, o_ev, m_t, m_ev in diffs:
            print(f"DIFF {pid}: oracle={o_t} {json.dumps(o_ev)[:200]}")
            print(f"     ours={m_t} {json.dumps(m_ev)[:200]}")


from ux_analyzer.analysis.slop.patterns import patterns as _p  # noqa: E402

PATTERNS = _p()

if __name__ == "__main__":
    main()
