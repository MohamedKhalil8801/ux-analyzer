"""Build the slop-detection context from a VisualSnapshot + page metadata.

Mirrors what @slop-detect/core's page-side script hands each pattern: a
`visible` element list (visibility-filtered), the first visible H1, color
helpers, viewport/document geometry, and the extracted text context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ux_analyzer.analysis.slop.color import Color, effective_background, parse_color
from ux_analyzer.analysis.visual.snapshot import Snapshot, SNode

STYLE_PROPS_EXTRA = [
    "background-image",
    "background-clip",
    "backdrop-filter",
    "-webkit-backdrop-filter",
    "filter",
    "font-style",
    "border-left-width",
    "border-right-width",
    "border-bottom-width",
    "border-top-color",
    "border-left-color",
    "border-right-color",
    "border-bottom-color",
    "grid-column",
    "grid-template-columns",
    "transform",
    "perspective",
]


def capture_meta_js() -> str:
    """Page-side JS returning viewport/doc geometry + text context.

    Returned object: ``{ viewport:{w,h}, docHeight, scrollY, textContext }``.
    ``textContext`` mirrors the reference `extractTextContext` (innerText of
    main content minus chrome, headings/paragraphs lists, word count).
    """
    return """
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
      return {
        text: text.slice(0, 200000),
        headings: headings,
        paragraphs: paragraphs,
        wordCount: words.length
      };
    }
    """


@dataclass(frozen=True, slots=True)
class SlopContext:
    """Everything the 27 design patterns read, precomputed from the snapshot."""

    snap: Snapshot
    visible: list[SNode] = field(default_factory=lambda: [])
    h1: SNode | None = None
    viewport_w: int = 1440
    viewport_h: int = 900
    doc_height: int = 0
    scroll_y: int = 0
    text_context: dict[str, Any] | None = None
    subtree_text: dict[int, str] = field(default_factory=lambda: {})
    surface: dict[str, str] = field(default_factory=lambda: {})

    def node_styles(self, node: SNode) -> dict[str, str]:
        return node.styles

    def parent_of(self, node: SNode) -> SNode | None:
        if node.parent < 0:
            return None
        return self.snap.nodes[node.parent]

    def full_text(self, node: SNode) -> str:
        """textContent approximation: own text + descendant texts (precomputed)."""
        return self.subtree_text.get(node.i, "")

    def effective_bg(self, node: SNode) -> Color | None:
        def styles_of(i: int) -> dict[str, str]:
            return self.snap.nodes[i].styles

        def parent_of(i: int) -> int | None:
            p = self.snap.nodes[i].parent
            return p if p >= 0 else None

        bg = effective_background(node.i, styles_of, parent_of)
        # The snapshot root is <body>; the reference walks live ancestors and
        # terminates at <html>. Mirror that by treating the html surface color
        # as the final ancestor before the white default.
        if bg == Color(255, 255, 255, 1.0):
            html_bg = parse_color(self.surface.get("htmlBg", ""))
            if html_bg and html_bg.a >= 0.5:
                return html_bg
        return bg


def _subtree_texts(snap: Snapshot) -> dict[int, str]:
    """Post-order accumulation of own text + descendant text per node."""
    out: dict[int, str] = {}
    by_parent: dict[int, list[SNode]] = {}
    for n in snap.nodes:
        by_parent.setdefault(n.parent, []).append(n)

    def fill(i: int) -> str:
        node = snap.nodes[i]
        parts = [(node.text or "").strip()]
        for child in by_parent.get(i, []):
            parts.append(fill(child.i))
        text = " ".join(p for p in parts if p)
        out[i] = text
        return text

    if snap.nodes:
        fill(0)
    return out


def _is_visible(node: SNode) -> bool:
    """Port of the reference `isVisible` (display/visibility/opacity/size)."""
    disp = node.style("display")
    if disp == "none":
        return False
    vis = node.style("visibility")
    if vis == "hidden":
        return False
    opacity = node.style("opacity")
    if opacity and opacity != "":
        try:
            if float(opacity) == 0.0:
                return False
        except ValueError:
            pass
    return node.box.w >= 4 and node.box.h >= 4


def build_context(
    snap: Snapshot,
    *,
    viewport_w: int = 1440,
    viewport_h: int = 900,
    doc_height: int = 0,
    scroll_y: int = 0,
    text_context: dict[str, Any] | None = None,
    surface: dict[str, str] | None = None,
) -> SlopContext:
    nodes = list(snap.nodes)
    visible = [n for n in nodes if _is_visible(n)]
    h1 = next((n for n in visible if n.tag == "h1"), None)
    return SlopContext(
        snap=snap,
        visible=visible,
        h1=h1,
        viewport_w=viewport_w,
        viewport_h=viewport_h,
        doc_height=doc_height or int(max((n.box.y + n.box.h for n in nodes), default=0)),
        scroll_y=scroll_y,
        text_context=text_context,
        subtree_text=_subtree_texts(snap),
        surface=dict(surface or {}),
    )
