"""Capture a VisualSnapshot from a rendered page via Playwright.

Works with any http(s) URL or local HTML file, so detectors generalize to
arbitrary sites. The snapshot format matches `benchmarks/ueye/evidence`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
    "background-image",
    "background-clip",
    "backdrop-filter",
    "-webkit-backdrop-filter",
    "filter",
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
    "border-right-color",
    "border-bottom-color",
    "border-left-color",
    "border-radius",
    "opacity",
    "width",
    "height",
    "box-shadow",
    "grid-column",
    "transform",
    "perspective",
]

_SNAPSHOT_JS = """
(props) => {
  const root = document.querySelector('[data-uxa-snapshot-root]') || document.body;
  const nodes = [];
  const walk = (el, parentIdx, depth) => {
    // Record every element (visible or not): the reference detector reads the
    // visible list for text/color scans but queries the LIVE DOM for icon
    // lookups (icon_card_grid, letter avatars), and those icons are often
    // zero-size or below-fold lazy images. Visibility filtering happens on the
    // analysis side. The cap bounds payload size on gigantic pages.
    if (nodes.length > 8000) return;
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
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

_META_JS = """
() => {
  function visibleText(root) {
    if (!root) return '';
    var t = root.innerText != null ? root.innerText : root.textContent;
    return (t || '').replace(/\\u00AD/g, '');
  }
  var main = document.querySelector('main, article, [role="main"]') || document.body;
  var clone = main.cloneNode(true);
  var strip = clone.querySelectorAll(
    'nav, footer, header, script, style, noscript, svg, code, pre, [aria-hidden="true"]'
  );
  for (var i = 0; i < strip.length; i++) {
    if (strip[i].parentNode) strip[i].parentNode.removeChild(strip[i]);
  }
  var text = visibleText(clone).trim();
  var headings = [];
  var hs = clone.querySelectorAll('h1, h2, h3, h4, li, dt');
  for (var j = 0; j < hs.length && headings.length < 200; j++) {
    var ht = (hs[j].innerText || hs[j].textContent || '').trim();
    if (ht) headings.push(ht.slice(0, 200));
  }
  var paragraphs = [];
  var ps = clone.querySelectorAll('p');
  for (var k = 0; k < ps.length && paragraphs.length < 200; k++) {
    var pt = (ps[k].innerText || ps[k].textContent || '').trim();
    if (pt) paragraphs.push(pt.slice(0, 400));
  }
  var words = text ? text.split(/\\s+/).filter(Boolean) : [];
  var centerEl = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2);
  return {
    viewport: { w: window.innerWidth, h: window.innerHeight },
    docHeight: document.documentElement.scrollHeight,
    scrollY: window.scrollY,
    surface: {
      htmlBg: getComputedStyle(document.documentElement).backgroundColor,
      bodyBg: getComputedStyle(document.body).backgroundColor,
      centerBg: centerEl ? getComputedStyle(centerEl).backgroundColor : ''
    },
    textContext: {
      text: text.slice(0, 200000),
      headings: headings,
      paragraphs: paragraphs,
      wordCount: words.length
    }
  };
}
"""

_SLOP_UA = "Mozilla/5.0 SlopDetector/1.0 (+https://github.com/ravidsrk/slop-detect)"

_DISABLE_JS = bool(os.environ.get("UXA_SLOP_DISABLE_JS"))


def extract_snapshot(
    source: str,
    viewport_width: int = 1280,
    viewport_height: int = 800,
) -> dict[str, Any]:
    """Render `source` (URL or file path) and return the snapshot dict.

    The dict carries ``rootBox``, ``nodes``, and page ``meta`` (viewport
    geometry, html-surface colors, and the extracted text context) consumed
    by the slop detector. Wait strategy and user agent mirror the reference
    slop-detect CLI (domcontentloaded + networkidle bounded + settle delay),
    so bot-gated pages resolve the same way. Set ``UXA_SLOP_DISABLE_JS=1`` to
    pin the DOM for deterministic benchmark comparisons against a frozen file.
    """
    path = Path(source)
    url = (
        "file:///" + str(path.resolve()).replace("\\", "/")
        if path.exists()
        else source
    )
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--allow-insecure-localhost"])
        ctx = b.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            user_agent=_SLOP_UA,
            java_script_enabled=not _DISABLE_JS,
        )
        pg = ctx.new_page()
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=45000)
            try:
                pg.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            pg.wait_for_timeout(500)
            raw = pg.evaluate(_SNAPSHOT_JS, _STYLE_PROPS)
            rb = pg.evaluate(
                """() => {
                  const r = document.querySelector('[data-uxa-snapshot-root], body')
                    .getBoundingClientRect();
                  return {x: r.x, y: r.y, w: r.width, h: r.height};
                }"""
            )
            meta = pg.evaluate(_META_JS)
        finally:
            b.close()
    return {"rootBox": rb, "meta": meta, **raw}


def extract_snapshot_to_file(source: str, out_path: str | Path) -> None:
    snap = extract_snapshot(source)
    Path(out_path).write_text(json.dumps(snap), encoding="utf-8")
