"""Rendered-page computed-style snapshot model.

A snapshot is a flat tree of nodes captured from the DOM of a rendered
page. Every node carries its bounding box and a fixed set of computed
styles. Snapshots are produced by `extract.py` (Playwright) and consumed by
the visual detectors; they serialize to plain JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class SNode:
    """One element in the snapshot tree."""

    i: int
    parent: int  # -1 for root
    depth: int
    tag: str
    cls: str
    id: str
    text: str  # direct text content only
    box: Box
    styles: dict[str, str]

    def style(self, prop: str) -> str:
        return self.styles.get(prop, "")

    def style_px(self, prop: str) -> float | None:
        """Parse a px length from a computed style value."""
        v = self.style(prop)
        if not v or v == "none" or v == "auto" or v == "normal":
            return None
        try:
            return float(v.removesuffix("px").strip())
        except ValueError:
            return None

    @property
    def classes(self) -> frozenset[str]:
        return frozenset(self.cls.split())


@dataclass(frozen=True, slots=True)
class Snapshot:
    root_box: Box
    nodes: tuple[SNode, ...]

    def children(self, i: int) -> list[SNode]:
        return [n for n in self.nodes if n.parent == i]

    def descendants(self, i: int) -> list[SNode]:
        out: list[SNode] = []
        stack = [i]
        while stack:
            cur = stack.pop()
            for n in self.nodes:
                if n.parent == cur:
                    out.append(n)
                    stack.append(n.i)
        return out

    def selector(self, node: SNode) -> str:
        """Stable human-readable path like 'div.card > ul > li.green[2]'."""
        parts: list[str] = []
        cur: SNode | None = node
        while cur is not None:
            # Build segment for cur with sibling index
            siblings = [n for n in self.nodes if n.parent == cur.parent and n.tag == cur.tag]
            if len(siblings) > 1:
                siblings_sorted = sorted(siblings, key=lambda n: n.i)
                idx = next((i for i, s in enumerate(siblings_sorted, 1) if s.i == cur.i), 1)
                # Use readable form tag:nth-of-type(idx) when needed
                seg = f"{cur.tag}:nth-of-type({idx})"
                # Add class hint for readability if present
                if cur.cls:
                    first_cls = cur.cls.split()[0]
                    seg = f"{cur.tag}.{first_cls}:nth-of-type({idx})"
            else:
                seg = cur.tag
                if cur.cls:
                    first_cls = cur.cls.split()[0]
                    seg = f"{cur.tag}.{first_cls}"
            parts.append(seg)
            cur = self.nodes[cur.parent] if cur.parent >= 0 else None
            if len(parts) >= 8:
                break
        parts.reverse()
        return " > ".join(parts)

    def css_selector(self, node: SNode) -> str:
        """Generate a unique CSS selector for programmatic lookup.

        Uses id if available, otherwise tag + classes + nth-of-type.
        The selector is suitable for `document.querySelector` and
        Playwright's `page.locator`.
        """
        parts: list[str] = []
        cur: SNode | None = node
        while cur is not None:
            if cur.id:
                segment = f"{cur.tag}#{cur.id}"
                parts.append(segment)
                break
            cls_part = ""
            if cur.cls:
                classes = [c for c in cur.cls.split() if c][:2]
                if classes:
                    cls_part = "." + ".".join(classes)
            segment = f"{cur.tag}{cls_part}"
            if cur.parent >= 0:
                siblings = [n for n in self.nodes if n.parent == cur.parent and n.tag == cur.tag]
                if len(siblings) > 1:
                    siblings_sorted = sorted(siblings, key=lambda n: n.i)
                    idx = next((i for i, s in enumerate(siblings_sorted, 1) if s.i == cur.i), 1)
                    segment += f":nth-of-type({idx})"
            parts.append(segment)
            cur = self.nodes[cur.parent] if cur.parent >= 0 else None
            if len(parts) >= 4:
                break
        parts.reverse()
        return " > ".join(parts)

    def xpath(self, node: SNode) -> str:
        """Generate an absolute XPath for programmatic lookup.

        If the node has an id, returns `//*[@id='...']` shortcut.
        Otherwise builds `/html/body/.../tag[n]` path.
        """
        if node.id:
            return f"//*[@id='{node.id}']"
        parts: list[str] = []
        cur: SNode | None = node
        while cur is not None:
            if cur.parent < 0:
                parts.append(f"/{cur.tag}")
            else:
                siblings = [n for n in self.nodes if n.parent == cur.parent and n.tag == cur.tag]
                if len(siblings) == 1:
                    parts.append(f"/{cur.tag}")
                else:
                    siblings_sorted = sorted(siblings, key=lambda n: n.i)
                    idx = next((i for i, s in enumerate(siblings_sorted, 1) if s.i == cur.i), 1)
                    parts.append(f"/{cur.tag}[{idx}]")
            cur = self.nodes[cur.parent] if cur.parent >= 0 else None
            if len(parts) > 20:
                break
        parts.reverse()
        xpath = "".join(parts)
        if not xpath.startswith("/"):
            xpath = "/" + xpath
        if not xpath.startswith("/html"):
            xpath = "/html" + xpath
        return xpath


def load_snapshot_json(path: str | Path) -> Snapshot:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return snapshot_from_dict(raw)


def snapshot_from_dict(raw: dict[str, Any]) -> Snapshot:
    rb = raw["rootBox"]
    nodes = tuple(
        SNode(
            i=n["i"],
            parent=n["parent"],
            depth=n["depth"],
            tag=n["tag"],
            cls=n.get("cls", ""),
            id=n.get("id", ""),
            text=n.get("text", ""),
            box=Box(
                x=float(n["box"]["x"]),
                y=float(n["box"]["y"]),
                w=float(n["box"]["w"]),
                h=float(n["box"]["h"]),
            ),
            styles=dict(n.get("styles", {})),
        )
        for n in raw["nodes"]
    )
    return Snapshot(
        root_box=Box(
            x=float(rb["x"]), y=float(rb["y"]), w=float(rb["w"]), h=float(rb["h"])
        ),
        nodes=nodes,
    )
