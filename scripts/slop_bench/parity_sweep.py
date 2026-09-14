"""Same-session parity sweep across the whole corpus (identical DOM per site)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from corpus import CORPUS  # noqa: E402

from ux_analyzer.analysis.slop.pipeline import analyze_slop  # noqa: E402
from ux_analyzer.analysis.visual.snapshot import snapshot_from_dict  # noqa: E402


def main() -> None:
    for entry in CORPUS:
        slug, url = entry["slug"], entry["url"]
        out = ROOT / "compare" / f"same_session_{slug}.json"
        proc = subprocess.run(
            ["node", str(ROOT / "ref_runner" / "same_session.mjs"), url, str(out)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(ROOT / "ref_runner"),
        )
        if proc.returncode != 0 or not out.is_file():
            print(f"{slug:16} CAPTURE FAIL {proc.stderr[-120:]}")
            continue
        data = json.loads(out.read_text(encoding="utf-8"))
        meta = data["meta"]
        ours = analyze_slop(
            snapshot_from_dict(data["snapshot"]),
            viewport_w=int(meta["viewport"]["w"]),
            viewport_h=int(meta["viewport"]["h"]),
            doc_height=int(meta.get("docHeight") or 0),
            scroll_y=int(meta.get("scrollY") or 0),
            text_context=meta.get("textContext"),
            surface=meta.get("surface"),
        )
        o = data["oracle"]["signals"]
        diffs = [
            p["id"]
            for p in ours["patterns"]
            if bool(o.get(p["id"], {}).get("triggered")) != p["triggered"]
        ]
        status = "PARITY" if not diffs else "DIFF " + ",".join(diffs)
        print(f"{slug:16} score={ours['score']} {status}")


if __name__ == "__main__":
    main()
