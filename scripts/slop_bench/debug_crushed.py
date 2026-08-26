"""Debug crushed_tracking diff on a same-session capture."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from ux_analyzer.analysis.slop.context import build_context  # noqa: E402
from ux_analyzer.analysis.slop.patterns import _crushed_tracking  # noqa: E402
from ux_analyzer.analysis.visual.snapshot import snapshot_from_dict  # noqa: E402

slug = sys.argv[1]
data = json.loads(
    (ROOT / "compare" / f"same_session_{slug}.json").read_text(encoding="utf-8")
)
print("oracle crushed:", json.dumps(data["oracle"]["signals"].get("crushed_tracking"))[:400])
meta = data["meta"]
ctx = build_context(
    snapshot_from_dict(data["snapshot"]),
    viewport_w=int(meta["viewport"]["w"]),
    viewport_h=int(meta["viewport"]["h"]),
    doc_height=int(meta.get("docHeight") or 0),
    scroll_y=int(meta.get("scrollY") or 0),
    text_context=meta.get("textContext"),
    surface=meta.get("surface"),
)
print("ours crushed:", _crushed_tracking(ctx))
print("--- candidates (font-size >= 28, non-normal letter-spacing) ---")
lead = re.compile(r"^[-+]?(\d+(?:\.\d+)?)")
for el in ctx.visible:
    fs = el.styles.get("font-size", "")
    ls = el.styles.get("letter-spacing", "")
    if ls in ("", "normal"):
        continue
    mfs = lead.match(fs)
    if not mfs:
        continue
    fsv = float(mfs.group(1))
    if fsv < 28:
        continue
    mls = lead.match(ls)
    lsv = float(mls.group(1)) if mls else 0.0
    txt = ctx.full_text(el)
    if len(txt) < 3 or len(txt) > 80:
        continue
    print(
        el.i, el.tag, el.cls[:30], "fs", fs, "ls", ls, "em", round(lsv / fsv, 4),
        "txt", txt[:25],
    )
