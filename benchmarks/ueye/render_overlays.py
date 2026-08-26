"""Render detector findings as red-box overlays on fixture screenshots."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from ux_analyzer.analysis.visual.snapshot import (  # noqa: E402
    Snapshot,
    load_snapshot_json,
)


def _resolve_nodes(snapshot: Snapshot, ref: str) -> list:
    """Match a selector-ish ref ('ul > li.green[i]') back to nodes."""
    out = []
    for n in snapshot.nodes:
        tag_ok = f"{n.tag}[" in ref or n.tag in ref.split(">")
        cls_ok = not n.cls or any(c in ref for c in n.classes)
        if tag_ok and cls_ok:
            out.append(n)
    return out


def render(base: pathlib.Path, slug: str, issues: list[dict], out_png: pathlib.Path) -> None:
    img_path = base / "evidence" / f"{slug}.png"
    if not img_path.exists():
        return
    img = Image.open(img_path).convert("RGB")
    snap = load_snapshot_json(base / "evidence" / f"{slug}.styles.json")
    root_x, root_y = snap.root_box.x, snap.root_box.y
    draw = ImageDraw.Draw(img)
    for issue in issues:
        for ref in issue.get("elements", [])[:8]:
            nodes = _resolve_nodes(snap, ref)
            for n in nodes[:4]:
                x0 = n.box.x - root_x
                y0 = n.box.y - root_y
                draw.rectangle(
                    [x0, y0, x0 + n.box.w, y0 + n.box.h],
                    outline=(220, 30, 30),
                    width=3,
                )
    img.save(out_png)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="benchmarks/ueye")
    args = ap.parse_args()
    base = pathlib.Path(args.base)
    report = json.loads((base / "reports" / "eval.json").read_text(encoding="utf-8"))
    out_dir = base / "reports" / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug, data in report["per_fixture"].items():
        render(base, slug, data["issues"], out_dir / f"{slug}.png")
        print(f"overlay {slug}: {len(data['issues'])} issues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
