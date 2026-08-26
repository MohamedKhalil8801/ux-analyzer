"""Capture a VisualSnapshot from a rendered page via Playwright.

Works with any http(s) URL or local HTML file, so detectors generalize to
arbitrary sites. The snapshot format matches `benchmarks/ueye/evidence`.
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

_STYLE_PROPS = [
    "display",
    "position",
    "flex-direction",
    "justify-content",
    "align-items",
    "gap",
    "row-gap",
    "column-gap",
    "grid-template-columns",
    "font-family",
    "font-size",
    "font-weight",
    "font-style",
    "line-height",
    "letter-spacing",
    "text-transform",
    "text-align",
    "text-decoration-line",
    "color",
    "background-color",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "border-top-width",
    "border-right-width",
    "border-bottom-width",
    "border-left-width",
    "border-top-color",
    "border-radius",
    "opacity",
    "width",
    "height",
    "box-shadow",
]

_SNAPSHOT_JS = """
(props) => {
  const root = document.querySelector('[data-uxa-snapshot-root]') || document.body;
  const nodes = [];
  const walk = (el, parentIdx, depth) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0 && el !== root) return;
    if (depth > 24 || nodes.length > 4000) return;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    const idx = nodes.length;
    const styles = {};
    for (const p of props) { styles[p] = cs.getPropertyValue(p); }
    let ownText = '';
    for (const n of el.childNodes) {
      if (n.nodeType === Node.TEXT_NODE) ownText += n.textContent;
    }
    nodes.push({
      i: idx,
      parent: parentIdx,
      depth,
      tag: el.tagName.toLowerCase(),
      cls: (typeof el.className === 'string') ? el.className : '',
      id: el.id || '',
      text: ownText.trim().slice(0, 300),
      box: {
        x: Math.round(r.x * 100) / 100,
        y: Math.round(r.y * 100) / 100,
        w: Math.round(r.width * 100) / 100,
        h: Math.round(r.height * 100) / 100,
      },
      styles,
    });
    for (const c of el.children) walk(c, idx, depth + 1);
  };
  walk(root, -1, 0);
  return { nodes };
}
"""


def extract_snapshot(
    source: str,
    viewport_width: int = 1440,
    viewport_height: int = 900,
) -> dict:
    """Render `source` (URL or file path) and return the snapshot dict."""
    path = Path(source)
    url = (
        "file:///" + str(path.resolve()).replace("\\", "/")
        if path.exists()
        else source
    )
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": viewport_width, "height": viewport_height})
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=45000)
            pg.wait_for_timeout(1200)
            raw = pg.evaluate(_SNAPSHOT_JS, _STYLE_PROPS)
            rb = pg.evaluate(
                """() => {
                  const r = document.querySelector('[data-uxa-snapshot-root], body')
                    .getBoundingClientRect();
                  return {x: r.x, y: r.y, w: r.width, h: r.height};
                }"""
            )
        finally:
            b.close()
    return {"rootBox": rb, **raw}


def extract_snapshot_to_file(source: str, out_path: str | Path) -> None:
    snap = extract_snapshot(source)
    Path(out_path).write_text(json.dumps(snap), encoding="utf-8")
