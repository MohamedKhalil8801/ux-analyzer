"""Freeze corpus URLs to rendered HTML so both detectors see identical DOM.

Usage: python scripts/slop_bench/freeze.py [slug ...]
Writes scripts/slop_bench/frozen/<slug>.html (full serialized DOM after the
same wait strategy both detectors use). Benchmarks then run the oracle and
our detector against the frozen file — eliminating live-page drift.

Note: serialized DOM only, so canvas-drawn content is lost; gradients/fonts
and computed styles come from the live CSS, which the frozen file still
references by absolute URL.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from corpus import CORPUS  # noqa: E402

FROZEN_DIR = ROOT / "frozen"


def freeze(slug: str, url: str, force: bool = False) -> Path:
    out = FROZEN_DIR / f"{slug}.html"
    if out.is_file() and not force:
        return out
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 SlopDetector/1.0 (+https://github.com/ravidsrk/slop-detect)",
            device_scale_factor=1,
        )
        pg = ctx.new_page()
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=45000)
            try:
                pg.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            pg.wait_for_timeout(500)
            html = pg.content()
        finally:
            b.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"frozen: {slug} {len(html)} bytes")
    return out


def main() -> None:
    slugs = sys.argv[1:] or [c["slug"] for c in CORPUS]
    by_slug = {c["slug"]: c for c in CORPUS}
    for slug in slugs:
        entry = by_slug[slug]
        freeze(entry["slug"], entry["url"])


if __name__ == "__main__":
    main()
