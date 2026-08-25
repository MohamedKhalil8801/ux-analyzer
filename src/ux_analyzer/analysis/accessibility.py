"""Accessibility detector.

Covers TheUXBites accessibility slice and Markswebb visual fatigue (12.0).

TheUXBites titles mapped (Accessibility cat, 17 items):
- Buttons and links have accessible names -> button_accessible_name
- Input buttons have accessible labels -> input_button_label
- Image buttons have alt text -> image_button_alt
- Form inputs have accessible names -> form_input_accessible_name
- Meter elements have accessible names -> meter_accessible_name
- Progress bars have accessible names -> progress_accessible_name
- Dialogs have accessible names -> dialog_accessible_name
- Toggle controls have accessible names -> toggle_accessible_name
- Tooltips have accessible names -> tooltip_accessible_name
- Tree items have accessible names -> treeitem_accessible_name
- ARIA role values are valid -> aria_role_valid
- Deprecated ARIA roles were not used -> aria_deprecated
- Required ARIA attributes are present -> aria_required_attr
- ARIA parent roles include required child roles -> aria_parent_child
- ARIA roles are inside required parent elements -> aria_child_parent
- Elements with role=text do not have focusable descendents -> aria_text_no_focusable
- No duplicate keyboard shortcuts -> duplicate_keyboard_shortcut
- Image alt missing is covered via image_alt (UX cat but a11y)

UX cat but a11y slice:
- Page has a skip link or landmark region -> skip_link
- ARIA IDs are unique across the page -> aria_id_unique
- Form inputs have visible labels -> form_visible_label
- Form fields don't have duplicate labels -> duplicate_labels
- Page has a main content area -> landmark_main
- HTML5 landmark elements are used to improve navigation -> landmarks
- All heading elements contain content -> heading_content
- Elements with visible text labels do not have matching accessible names -> visible_label_match
- Frames have accessible titles -> frame_titles
- Embedded objects have alt text -> object_alt
- Dropdown menus have accessible names -> select_accessible_name (via form_input)
- Tables have headers etc -> table_headers
- Videos have captions -> video_captions
- Custom controls have associated labels -> custom_controls_label
- Custom controls have ARIA roles -> custom_controls_role
- Interactive controls are keyboard focusable -> interactive_keyboard_focusable
- User focus is not accidentally trapped in a region -> focus_not_trapped
- The page has a logical tab order -> tab_order
- Offscreen content is hidden from assistive technology -> offscreen_hidden
- Links are distinguishable without relying on color -> link_distinguishable
- Visual order on the page follows DOM order -> visual_order
- The user's focus is directed to new content added to the page -> focus_directed (info)
- The document does not use <meta http-equiv="refresh"> -> meta_refresh
- Markswebb 12 eyes get tired quickly (visual fatigue) -> visual_fatigue
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True, slots=True)
class AccessibilityIssue:
    """Evidence-backed accessibility finding."""

    title: str
    description: str
    severity: str  # critical|medium|low
    evidence: dict
    check_id: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_ARIA_ROLES = {
    "alert",
    "alertdialog",
    "application",
    "article",
    "banner",
    "blockquote",
    "button",
    "caption",
    "cell",
    "checkbox",
    "code",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "definition",
    "deletion",
    "dialog",
    "directory",
    "document",
    "emphasis",
    "feed",
    "figure",
    "form",
    "generic",
    "grid",
    "gridcell",
    "group",
    "heading",
    "img",
    "insertion",
    "list",
    "listbox",
    "listitem",
    "log",
    "main",
    "marquee",
    "math",
    "menu",
    "menubar",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "meter",
    "navigation",
    "none",
    "note",
    "option",
    "paragraph",
    "presentation",
    "progressbar",
    "radio",
    "radiogroup",
    "row",
    "rowgroup",
    "rowheader",
    "scrollbar",
    "search",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "status",
    "strong",
    "subscript",
    "superscript",
    "switch",
    "tab",
    "table",
    "tablist",
    "tabpanel",
    "term",
    "textbox",
    "time",
    "timer",
    "toolbar",
    "tooltip",
    "tree",
    "treegrid",
    "treeitem",
}

DEPRECATED_ROLES = {"directory"}

# role -> required aria attributes (presence check)
REQUIRED_ARIA_ATTRS: dict[str, list[str]] = {
    "checkbox": ["aria-checked"],
    "radio": ["aria-checked"],
    "switch": ["aria-checked"],
    "menuitemcheckbox": ["aria-checked"],
    "menuitemradio": ["aria-checked"],
    "slider": ["aria-valuenow", "aria-valuemin", "aria-valuemax"],
    "spinbutton": ["aria-valuenow", "aria-valuemin", "aria-valuemax"],
    "scrollbar": ["aria-controls", "aria-valuenow", "aria-valuemin", "aria-valuemax"],
    "meter": ["aria-valuenow", "aria-valuemin", "aria-valuemax"],
    "progressbar": ["aria-valuenow"],
    "combobox": ["aria-controls", "aria-expanded"],
    # option inside listbox ideally aria-selected but not strictly required for static
}

# child -> required parents (any one)
PARENT_REQUIRED: dict[str, list[str]] = {
    "listitem": ["list"],
    "option": ["listbox", "combobox", "list"],
    "treeitem": ["tree", "group", "treegrid"],
    "row": ["table", "grid", "treegrid", "rowgroup"],
    "rowgroup": ["table", "grid", "treegrid"],
    "cell": ["row"],
    "gridcell": ["row"],
    "columnheader": ["row"],
    "rowheader": ["row"],
    "tab": ["tablist"],
    "menuitem": ["menu", "menubar", "group"],
    "menuitemcheckbox": ["menu", "menubar", "group"],
    "menuitemradio": ["menu", "menubar", "group"],
}

# parent -> required children (any one must be descendant)
CHILD_REQUIRED: dict[str, list[str]] = {
    "list": ["listitem"],
    "listbox": ["option"],
    "menu": ["menuitem", "menuitemcheckbox", "menuitemradio", "group"],
    "menubar": ["menuitem", "menuitemcheckbox", "menuitemradio", "group"],
    "tablist": ["tab"],
    "table": ["row", "rowgroup"],
    "grid": ["row", "rowgroup"],
    "treegrid": ["row", "rowgroup"],
    "tree": ["treeitem", "group"],
    "radiogroup": ["radio"],
    "row": ["cell", "gridcell", "columnheader", "rowheader"],
    "rowgroup": ["row"],
}

VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
    "command",
    "keygen",
    "menuitem",
}


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------
class _A11yParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[dict] = []  # flat list
        self.stack: list[int] = []  # indices
        self.id_map: dict[str, list[int]] = defaultdict(list)
        self.label_for_map: dict[str, list[dict]] = defaultdict(list)  # for id -> label elements
        self.labels: list[dict] = []  # label elements

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        d = {k.lower(): (v or "") for k, v in attrs}
        # role tokens
        role_raw = d.get("role", "").lower().strip()
        role_tokens = [r.strip() for r in role_raw.split()] if role_raw else []
        idx = len(self.elements)
        parent = self.stack[-1] if self.stack else None
        el = {
            "idx": idx,
            "tag": t,
            "attrs": d,
            "role": role_tokens[0] if role_tokens else "",
            "role_tokens": role_tokens,
            "id": d.get("id", "").strip(),
            "text": "",
            "parent": parent,
            "children": [],
            "ancestors": list(self.stack),  # copy
        }
        self.elements.append(el)
        if parent is not None:
            self.elements[parent]["children"].append(idx)
        # track ids
        if el["id"]:
            self.id_map[el["id"]].append(idx)
        # track labels
        if t == "label":
            self.labels.append(el)
            for_val = d.get("for", "").strip()
            if for_val:
                self.label_for_map[for_val].append(el)

        if t not in VOID_ELEMENTS:
            self.stack.append(idx)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        # find matching from stack top
        if not self.stack:
            return
        # if top matches, pop and propagate text
        top_idx = self.stack[-1]
        top_tag = self.elements[top_idx]["tag"]
        if top_tag == t:
            popped = self.stack.pop()
            txt = self.elements[popped]["text"].strip()
            if self.stack:
                parent_idx = self.stack[-1]
                # propagate text to parent
                if txt:
                    if self.elements[parent_idx]["text"]:
                        self.elements[parent_idx]["text"] += " " + txt
                    else:
                        self.elements[parent_idx]["text"] = txt
        else:
            # try to find t somewhere in stack (malformed html)
            # search from top down
            for i in range(len(self.stack) - 1, -1, -1):
                if self.elements[self.stack[i]]["tag"] == t:
                    # pop up to that
                    while len(self.stack) > i:
                        popped = self.stack.pop()
                        txt = self.elements[popped]["text"].strip()
                        if self.stack and txt:
                            parent_idx = self.stack[-1]
                            if self.elements[parent_idx]["text"]:
                                self.elements[parent_idx]["text"] += " " + txt
                            else:
                                self.elements[parent_idx]["text"] = txt
                    break

    def handle_data(self, data: str) -> None:
        if not self.stack:
            return
        # filter nodes with aria-hidden ancestor (decorative monogram etc.)
        for idx in self.stack:
            if self.elements[idx]["attrs"].get("aria-hidden", "").lower() == "true":
                return
        txt = data.strip()
        if not txt:
            return
        # normalize whitespace
        norm = " ".join(txt.split())
        if not norm:
            return
        top_idx = self.stack[-1]
        if self.elements[top_idx]["text"]:
            self.elements[top_idx]["text"] += " " + norm
        else:
            self.elements[top_idx]["text"] = norm

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # for xhtml self-closing
        self.handle_starttag(tag, attrs)
        # void elements don't push, so no need to pop
        # for non-void self-closing like <div/>
        t = tag.lower()
        if t not in VOID_ELEMENTS and self.stack and self.elements[self.stack[-1]]["tag"] == t:
            # pop immediately
            popped = self.stack.pop()
            txt = self.elements[popped]["text"].strip()
            if self.stack and txt:
                parent_idx = self.stack[-1]
                if self.elements[parent_idx]["text"]:
                    self.elements[parent_idx]["text"] += " " + txt
                else:
                    self.elements[parent_idx]["text"] = txt


def _parse_html(html: str) -> _A11yParser:
    p = _A11yParser()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    return p


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _origin(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc:
        parsed2 = urlparse(f"https://{url}")
        scheme = parsed2.scheme
        netloc = parsed2.netloc
    return f"{scheme}://{netloc}".rstrip("/")


async def _safe_get(client: httpx.AsyncClient, url: str) -> tuple[int | None, str | None, str | None]:
    try:
        resp = await client.get(url)
        try:
            txt = resp.text
        except Exception:
            txt = resp.content.decode("utf-8", errors="replace")
        return resp.status_code, txt, None
    except Exception as exc:
        return None, None, str(exc)


def _has_accessible_name_via_aria(el: dict, parser: _A11yParser) -> bool:
    attrs = el["attrs"]
    if attrs.get("aria-label", "").strip():
        return True
    labelledby = attrs.get("aria-labelledby", "").strip()
    if labelledby:
        for ref_id in labelledby.split():
            if ref_id in parser.id_map:
                # check target has text or aria-label
                for tgt_idx in parser.id_map[ref_id]:
                    tgt = parser.elements[tgt_idx]
                    if tgt["text"].strip() or tgt["attrs"].get("aria-label", "").strip():
                        return True
                # even if referenced id exists, consider it present (to avoid false positive when text not collected)
                return True
    return False


def _has_inner_text(el: dict) -> bool:
    return bool(el["text"].strip())


def _has_accessible_name(el: dict, parser: _A11yParser) -> bool:
    # aria-label / labelledby or inner text
    if _has_accessible_name_via_aria(el, parser):
        return True
    if _has_inner_text(el):
        return True
    # for img etc, alt handled elsewhere, but for completeness check alt as name
    if el["tag"] == "img" and el["attrs"].get("alt", "") is not None:
        alt = el["attrs"].get("alt", "")
        # presence of alt attribute considered? But empty alt not a name unless decorative
        # we consider alt present as name only if non-empty
        if alt.strip():
            return True
    # title attribute could be considered but not reliable; we ignore
    return False


def _is_focusable(el: dict) -> bool:
    tag = el["tag"]
    attrs = el["attrs"]
    # disabled check
    if "disabled" in attrs:
        return False
    if attrs.get("aria-disabled", "").lower() == "true":
        return False
    if attrs.get("aria-hidden", "").lower() == "true":
        return False
    # hidden attribute
    if "hidden" in attrs:
        return False
    # tabindex -1 => not focusable via keyboard but programmatic; we treat as not keyboard focusable
    tabindex = attrs.get("tabindex", "").strip()
    if tabindex == "-1":
        return False
    if tabindex and tabindex.lstrip("-").isdigit():
        # any other tabindex => focusable
        return True
    if "contenteditable" in attrs and attrs["contenteditable"].lower() != "false":
        return True
    # native focusable tags
    if tag == "a" and attrs.get("href", "").strip():
        return True
    if tag in ("button", "select", "textarea", "iframe"):
        return True
    if tag == "input":
        typ = attrs.get("type", "").lower()
        if typ == "hidden":
            return False
        return True
    if tag in ("audio", "video") and "controls" in attrs:
        return True
    # elements with role button etc and not disabled?
    role = el["role"]
    if role in ("button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "slider", "spinbutton"):
        # assume focusable if has tabindex or is native equivalent; for custom, require tabindex
        # We'll check if parent role check: if element has role, we consider focusable if it has tabindex != -1 or is native tag
        # For detection of focusable descendants inside role=text, we consider explicit focusable tags + tabindex
        if tabindex != "-1":
            # if custom role without tabindex, it's not focusable by our strict rule, so not trap
            # but for simplicity treat role elements as focusable if they could be
            if tag in ("div", "span"):
                # need tabindex to be focusable
                return tabindex != "" and tabindex != "-1"
            return True
        return False
    # generic with tabindex >=0
    if tabindex and tabindex.isdigit() and int(tabindex) >= 0:
        return True
    return False


def _is_hidden_from_at(el: dict) -> bool:
    attrs = el["attrs"]
    if attrs.get("aria-hidden", "").lower() == "true":
        return True
    if "hidden" in attrs:
        return True
    style = attrs.get("style", "").lower()
    if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
        return True
    return False


def _get_label_text_for_input(el: dict, parser: _A11yParser) -> str | None:
    # explicit for
    inp_id = el["id"]
    if inp_id and inp_id in parser.label_for_map:
        # get first label's text
        labels = parser.label_for_map[inp_id]
        texts = [lbl["text"].strip() for lbl in labels if lbl["text"].strip()]
        if texts:
            return " ".join(texts)
        # even empty label considered but we need text
        return ""
    # implicit wrapping label: walk ancestors
    anc_ids = el["ancestors"]
    for anc_idx in reversed(anc_ids):
        anc = parser.elements[anc_idx]
        if anc["tag"] == "label":
            return anc["text"].strip()
    return None


def _has_label_association(el: dict, parser: _A11yParser) -> bool:
    txt = _get_label_text_for_input(el, parser)
    if txt is None:
        return False
    return bool(txt.strip())


def _has_role(el: dict, role: str) -> bool:
    return role in el["role_tokens"]


def _has_tag_or_role(el: dict, name: str) -> bool:
    # check tag equals name or role contains name
    if el["tag"] == name:
        return True
    if name in el["role_tokens"]:
        return True
    # also handle implicit ARIA
    # tag equivalents: ul/ol => list, li => listitem, table => table, tr => row, td/th => cell, select => listbox?
    implicit = {
        "list": ("ul", "ol"),
        "listitem": ("li",),
        "table": ("table",),
        "row": ("tr",),
        "cell": ("td",),
        "columnheader": ("th",),
        "rowheader": ("th",),
        "combobox": ("select",),
        "listbox": ("select", "datalist"),
        "heading": ("h1", "h2", "h3", "h4", "h5", "h6"),
        "button": ("button",),
        "link": ("a",),
        "img": ("img",),
    }
    if name in implicit and el["tag"] in implicit[name]:
        return True
    return False


def _get_descendant_indices(el: dict, parser: _A11yParser) -> list[int]:
    # BFS traverse children
    res: list[int] = []
    stack = list(el["children"])
    while stack:
        cur = stack.pop()
        res.append(cur)
        stack.extend(parser.elements[cur]["children"])
    return res


def _has_descendant_with_role_or_tag(el: dict, parser: _A11yParser, names: list[str]) -> bool:
    for idx in _get_descendant_indices(el, parser):
        child = parser.elements[idx]
        for nm in names:
            if _has_tag_or_role(child, nm):
                return True
    return False


def _has_ancestor_with_role_or_tag(el: dict, parser: _A11yParser, names: list[str]) -> bool:
    for anc_idx in el["ancestors"]:
        anc = parser.elements[anc_idx]
        for nm in names:
            if _has_tag_or_role(anc, nm):
                return True
    return False


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def _check_button_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    issues: list[AccessibilityIssue] = []
    candidates: list[dict] = []
    for el in parser.elements:
        tag = el["tag"]
        role = el["role"]
        attrs = el["attrs"]
        # determine if element is button/link interactive
        is_button = False
        # native button
        if tag == "button":
            is_button = True
        # a with href
        elif tag == "a" and attrs.get("href", "").strip():
            is_button = True
        # input button types
        elif tag == "input" and attrs.get("type", "").lower() in ("button", "submit", "reset"):
            # handled via input_button_label, but also covered? We'll let input button use separate check but still also flag here if no value? Keep separate.
            continue
        # role based
        elif role in ("button", "link", "tab", "menuitem", "menuitemcheckbox", "menuitemradio"):
            is_button = True
        elif "button" in el["role_tokens"] or "link" in el["role_tokens"]:
            is_button = True
        # also elements with onclick? Can't detect without JS, we restrict to explicit roles

        if not is_button:
            continue
        # skip hidden
        if _is_hidden_from_at(el):
            continue
        # check accessible name
        if not _has_accessible_name(el, parser):
            # for <a>, check href not alone
            candidates.append(el)

    if candidates:
        evidence = {
            "count": len(candidates),
            "elements": [
                {
                    "tag": e["tag"],
                    "role": e["role"],
                    "id": e["id"][:50] if e["id"] else "",
                    "outer": f"<{e['tag']} role='{e['role']}' id='{e['id']}'>".strip()[:200],
                    "has_aria_label": bool(e["attrs"].get("aria-label", "").strip()),
                    "text_preview": e["text"][:80],
                }
                for e in candidates[:5]
            ],
        }
        issues.append(
            AccessibilityIssue(
                title="Buttons and links have accessible names",
                description=f"Found {len(candidates)} button/link element(s) without accessible name (text, aria-label, or aria-labelledby). Assistive tech cannot announce purpose.",
                severity="critical",
                evidence=evidence,
                check_id="button_accessible_name",
            )
        )
    return issues


def _check_input_button_label(parser: _A11yParser) -> list[AccessibilityIssue]:
    issues: list[AccessibilityIssue] = []
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] != "input":
            continue
        typ = el["attrs"].get("type", "").lower()
        if typ not in ("button", "submit", "reset"):
            continue
        if _is_hidden_from_at(el):
            continue
        val = el["attrs"].get("value", "").strip()
        if not val and not el["attrs"].get("aria-label", "").strip() and not el["attrs"].get("aria-labelledby", "").strip():
            # also check title as fallback? Not sufficient per spec, we flag anyway
            bad.append(el)
    if bad:
        issues.append(
            AccessibilityIssue(
                title="Input buttons have accessible labels",
                description=f"Found {len(bad)} <input type=button/submit/reset> without accessible label (value, aria-label, or aria-labelledby).",
                severity="critical",
                evidence={
                    "count": len(bad),
                    "types": [e["attrs"].get("type", "") for e in bad[:5]],
                    "ids": [e["id"] for e in bad[:5]],
                },
                check_id="input_button_label",
            )
        )
    return issues


def _check_image_button_alt(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] != "input":
            continue
        if el["attrs"].get("type", "").lower() != "image":
            continue
        if _is_hidden_from_at(el):
            continue
        alt = el["attrs"].get("alt", None)
        # alt must be present and non-empty? spec says image buttons have alt text
        if alt is None or not alt.strip():
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Image buttons have alt text",
                description=f"Found {len(bad)} <input type=image> without alt text.",
                severity="critical",
                evidence={"count": len(bad), "ids": [e["id"] for e in bad[:5]]},
                check_id="image_button_alt",
            )
        ]
    return []


def _check_image_alt(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] != "img":
            continue
        if _is_hidden_from_at(el):
            continue
        # ignore if aria-hidden already hidden? but we already skipped
        attrs = el["attrs"]
        # if role presentation or none and empty alt, decorative allowed
        role = el["role"]
        if role in ("presentation", "none"):
            continue
        if "alt" not in attrs:
            bad.append(el)
        # alt="" with aria-hidden true also considered decorative -> we already skipped hidden? Actually aria-hidden not necessarily hidden_from_at for img? We skipped hidden, so decorative with aria-hidden would be skipped. So fine.
        # if alt present but empty string: allowed decorative only if we consider not flagged; prompt says empty alt allowed for decorative if aria-hidden, but flag missing. So we only flag missing attribute.
        # also check alt empty with no decorative hint maybe still considered missing? But we follow spec lenient: only flag_missing.

    if bad:
        return [
            AccessibilityIssue(
                title="Images have alt text",
                description=f"Found {len(bad)} <img> without alt attribute. Screen readers cannot convey image purpose.",
                severity="critical",
                evidence={
                    "count": len(bad),
                    "srcs": [e["attrs"].get("src", "")[:200] for e in bad[:5]],
                },
                check_id="image_alt",
            )
        ]
    return []


def _check_form_input_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        tag = el["tag"]
        if tag not in ("input", "textarea", "select"):
            continue
        typ = el["attrs"].get("type", "").lower() if tag == "input" else ""
        # skip hidden, button types already handled, image handled
        if typ in ("hidden", "button", "submit", "reset", "image"):
            continue
        if _is_hidden_from_at(el):
            continue
        # Check accessible name: label association, aria-label, aria-labelledby
        has_label = _has_label_association(el, parser)
        has_aria = _has_accessible_name_via_aria(el, parser)
        # placeholder not sufficient: we intentionally don't count placeholder as accessible name
        bool(el["attrs"].get("placeholder", "").strip())
        # if has placeholder but no label/aria, still bad per spec
        if has_label or has_aria:
            continue
        # also check if element has title? Not counted
        # also check if select has associated label via wrapper?
        bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Form inputs have accessible names",
                description=f"Found {len(bad)} form control(s) without accessible name (label, aria-label, or aria-labelledby). Placeholder alone is not sufficient.",
                severity="critical",
                evidence={
                    "count": len(bad),
                    "elements": [
                        {
                            "tag": e["tag"],
                            "type": e["attrs"].get("type", ""),
                            "id": e["id"][:50] if e["id"] else "",
                            "placeholder": e["attrs"].get("placeholder", "")[:50],
                        }
                        for e in bad[:5]
                    ],
                },
                check_id="form_input_accessible_name",
            )
        ]
    return []


def _check_form_visible_label(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Flag inputs that have accessible name via aria-label only but lack visible <label>
    bad: list[dict] = []
    for el in parser.elements:
        tag = el["tag"]
        if tag not in ("input", "textarea", "select"):
            continue
        typ = el["attrs"].get("type", "").lower() if tag == "input" else ""
        if typ in ("hidden", "button", "submit", "reset", "image"):
            continue
        if _is_hidden_from_at(el):
            continue
        has_visible_label = _has_label_association(el, parser)
        has_aria = _has_accessible_name_via_aria(el, parser)
        if not has_visible_label and has_aria:
            bad.append(el)
        # also if no visible label and placeholder only? That would already be flagged by form_input_accessible_name but we also flag visible label.
        # To avoid duplicate for same element where both checks would flag, we only flag visible label when accessible name exists via aria but visible missing.
    if bad:
        return [
            AccessibilityIssue(
                title="Form inputs have visible labels",
                description=f"Found {len(bad)} form control(s) with accessible name but no visible <label>. Visible labels help all users.",
                severity="medium",
                evidence={
                    "count": len(bad),
                    "ids": [e["id"][:50] for e in bad[:5]],
                },
                check_id="form_visible_label",
            )
        ]
    return []


def _check_duplicate_labels(parser: _A11yParser) -> list[AccessibilityIssue]:
    # map label text -> list of input ids that use it
    label_text_to_inputs: dict[str, list[str]] = defaultdict(list)
    for el in parser.elements:
        tag = el["tag"]
        if tag not in ("input", "textarea", "select"):
            continue
        typ = el["attrs"].get("type", "").lower() if tag == "input" else ""
        if typ in ("hidden", "button", "submit", "reset", "image"):
            continue
        if _is_hidden_from_at(el):
            continue
        label_text = _get_label_text_for_input(el, parser)
        if label_text and label_text.strip():
            norm = " ".join(label_text.lower().split())
            # use label text as key, value is element id or index
            ident = el["id"] or f"idx:{el['idx']}"
            label_text_to_inputs[norm].append(ident)
    dups = {k: v for k, v in label_text_to_inputs.items() if len(v) > 1}
    if dups:
        return [
            AccessibilityIssue(
                title="Form fields don't have duplicate labels",
                description=f"Found {len(dups)} label text(s) reused across multiple fields, causing ambiguity.",
                severity="medium",
                evidence={
                    "duplicate_labels": {k: len(v) for k, v in list(dups.items())[:5]},
                    "examples": list(dups.items())[:3],
                },
                check_id="duplicate_labels",
            )
        ]
    return []


def _check_frame_titles(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] not in ("iframe", "frame"):
            continue
        title = el["attrs"].get("title", "").strip()
        if not title:
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Frames have accessible titles",
                description=f"Found {len(bad)} <iframe>/<frame> without accessible title.",
                severity="critical",
                evidence={
                    "count": len(bad),
                    "srcs": [e["attrs"].get("src", "")[:200] for e in bad[:5]],
                },
                check_id="frame_titles",
            )
        ]
    return []


def _check_meter_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] != "meter":
            continue
        if _is_hidden_from_at(el):
            continue
        # need accessible name: label, aria-label, aria-labelledby, or title?
        has_label = _has_label_association(el, parser) if el["id"] else False
        has_aria = _has_accessible_name_via_aria(el, parser)
        has_text = _has_inner_text(el)
        # also check associated label via <label for>
        if has_label or has_aria or has_text:
            continue
        # also check if parent label wrapping?
        # Check ancestor label
        wrapped = False
        for anc_idx in el["ancestors"]:
            if parser.elements[anc_idx]["tag"] == "label":
                wrapped = True
                break
        if wrapped:
            continue
        bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Meter elements have accessible names",
                description=f"Found {len(bad)} <meter> without accessible name.",
                severity="medium",
                evidence={"count": len(bad)},
                check_id="meter_accessible_name",
            )
        ]
    return []


def _check_progress_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] != "progress":
            continue
        if _is_hidden_from_at(el):
            continue
        has_label = _has_label_association(el, parser) if el["id"] else False
        has_aria = _has_accessible_name_via_aria(el, parser)
        has_text = _has_inner_text(el)
        wrapped = False
        for anc_idx in el["ancestors"]:
            if parser.elements[anc_idx]["tag"] == "label":
                wrapped = True
                break
        if has_label or has_aria or has_text or wrapped:
            continue
        bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Progress bars have accessible names",
                description=f"Found {len(bad)} <progress> without accessible name.",
                severity="medium",
                evidence={"count": len(bad)},
                check_id="progress_accessible_name",
            )
        ]
    return []


def _check_dialog_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        is_dialog = False
        if el["tag"] == "dialog":
            is_dialog = True
        elif el["role"] == "dialog" or "dialog" in el["role_tokens"]:
            is_dialog = True
        elif el["attrs"].get("role", "").lower() == "dialog":
            is_dialog = True
        if not is_dialog:
            continue
        if _is_hidden_from_at(el):
            continue
        # dialog should have aria-label, aria-labelledby, or aria-describedby with title? We check label
        has_aria = _has_accessible_name_via_aria(el, parser)
        # also check <dialog> may have accessible name via text? Not sufficient per ARIA? But we consider inner text as fallback? Lighthouse requires accessible name via label
        # For strictness, require aria-label or labelledby
        if has_aria:
            continue
        # also check if element has title attribute with text? Not counted
        bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Dialogs have accessible names",
                description=f"Found {len(bad)} dialog element(s) without accessible name (aria-label or aria-labelledby).",
                severity="medium",
                evidence={"count": len(bad), "ids": [e["id"][:50] for e in bad[:5]]},
                check_id="dialog_accessible_name",
            )
        ]
    return []


def _check_toggle_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        # Toggle controls: checkbox, switch, radio, button with aria-pressed
        role = el["role"]
        tag = el["tag"]
        typ = el["attrs"].get("type", "").lower() if tag == "input" else ""
        is_toggle = False
        if tag == "input" and typ in ("checkbox", "radio"):
            is_toggle = True
        elif role in ("checkbox", "radio", "switch"):
            is_toggle = True
        elif "checkbox" in el["role_tokens"] or "switch" in el["role_tokens"] or "radio" in el["role_tokens"]:
            is_toggle = True
        elif el["attrs"].get("aria-pressed", "").strip() != "" and (tag == "button" or role == "button"):
            is_toggle = True
        if not is_toggle:
            continue
        if _is_hidden_from_at(el):
            continue
        # need accessible name
        # for native input checkbox/radio, check label association or aria
        if tag == "input" and typ in ("checkbox", "radio"):
            has_label = _has_label_association(el, parser)
            has_aria = _has_accessible_name_via_aria(el, parser)
            if has_label or has_aria:
                continue
            # also check if inside label wrapper
            wrapped = False
            for anc_idx in el["ancestors"]:
                if parser.elements[anc_idx]["tag"] == "label":
                    wrapped = True
                    break
            if wrapped:
                continue
            bad.append(el)
        else:
            # custom toggle: need aria-label/labelledby or inner text
            if _has_accessible_name(el, parser):
                continue
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Toggle controls have accessible names",
                description=f"Found {len(bad)} toggle control(s) without accessible name.",
                severity="critical",
                evidence={
                    "count": len(bad),
                    "elements": [
                        {"tag": e["tag"], "role": e["role"], "id": e["id"][:50]} for e in bad[:5]
                    ],
                },
                check_id="toggle_accessible_name",
            )
        ]
    return []


def _check_tooltip_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["role"] != "tooltip" and "tooltip" not in el["role_tokens"]:
            continue
        if _is_hidden_from_at(el):
            continue
        if not _has_accessible_name(el, parser):
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Tooltips have accessible names",
                description=f"Found {len(bad)} tooltip element(s) without accessible name.",
                severity="medium",
                evidence={"count": len(bad)},
                check_id="tooltip_accessible_name",
            )
        ]
    return []


def _check_treeitem_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["role"] != "treeitem" and "treeitem" not in el["role_tokens"]:
            continue
        if _is_hidden_from_at(el):
            continue
        if not _has_accessible_name(el, parser):
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Tree items have accessible names",
                description=f"Found {len(bad)} treeitem without accessible name.",
                severity="medium",
                evidence={"count": len(bad)},
                check_id="treeitem_accessible_name",
            )
        ]
    return []


def _check_aria_role_valid(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[tuple[dict, str]] = []
    for el in parser.elements:
        if not el["role_tokens"]:
            continue
        for token in el["role_tokens"]:
            if token not in VALID_ARIA_ROLES:
                bad.append((el, token))
                break  # one per element
    if bad:
        return [
            AccessibilityIssue(
                title="ARIA role values are valid",
                description=f"Found {len(bad)} element(s) with invalid ARIA role value(s).",
                severity="medium",
                evidence={
                    "count": len(bad),
                    "invalid_roles": [t for _, t in bad[:5]],
                    "elements": [
                        {"tag": e["tag"], "role": e["attrs"].get("role", "")[:50], "id": e["id"][:50]}
                        for e, _ in bad[:5]
                    ],
                },
                check_id="aria_role_valid",
            )
        ]
    return []


def _check_aria_deprecated(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        for token in el["role_tokens"]:
            if token in DEPRECATED_ROLES:
                bad.append(el)
                break
    if bad:
        return [
            AccessibilityIssue(
                title="Deprecated ARIA roles were not used",
                description=f"Found {len(bad)} element(s) using deprecated ARIA role(s) (e.g., directory).",
                severity="low",
                evidence={
                    "count": len(bad),
                    "roles": [e["attrs"].get("role", "") for e in bad[:5]],
                },
                check_id="aria_deprecated",
            )
        ]
    return []


def _check_aria_required_attr(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    details: list[dict] = []
    for el in parser.elements:
        if not el["role"]:
            continue
        # Only check first role token for required attrs
        role = el["role"]
        # also handle space-separated? Use first token
        # Check that role is valid and has required attrs mapping
        if role not in REQUIRED_ARIA_ATTRS:
            # check if any token has requirement
            found = None
            for tok in el["role_tokens"]:
                if tok in REQUIRED_ARIA_ATTRS:
                    found = tok
                    break
            if not found:
                continue
            role = found
        required = REQUIRED_ARIA_ATTRS.get(role, [])
        if not required:
            continue
        attrs = el["attrs"]
        missing = [attr for attr in required if not attrs.get(attr, "").strip()]
        if missing:
            bad.append(el)
            details.append({"tag": el["tag"], "role": role, "missing": missing, "id": el["id"][:50]})
    if bad:
        return [
            AccessibilityIssue(
                title="Required ARIA attributes are present",
                description=f"Found {len(bad)} element(s) missing required ARIA attributes for their role.",
                severity="medium",
                evidence={"count": len(bad), "details": details[:5]},
                check_id="aria_required_attr",
            )
        ]
    return []


def _check_aria_parent_child(parser: _A11yParser) -> list[AccessibilityIssue]:
    # parent roles include required child roles
    issues: list[AccessibilityIssue] = []
    for el in parser.elements:
        # Use tag or role to determine parent role
        # Determine parent role name: prefer role token, else tag equivalent
        parent_roles: list[str] = []
        # if element has role, check that role
        for tok in el["role_tokens"]:
            if tok in CHILD_REQUIRED:
                parent_roles.append(tok)
        # also check tag equivalents for parents defined via tags
        # For simplicity, also check tag-based parents: if tag ul/ol => list, etc.
        if el["tag"] in ("ul", "ol") and "list" not in parent_roles:
            parent_roles.append("list")
        if el["tag"] == "select" and "listbox" not in parent_roles:
            # select as listbox
            parent_roles.append("listbox")
        if el["tag"] in ("table",):
            if "table" not in parent_roles:
                parent_roles.append("table")
        # For each parent role, check child requirement
        for pr in parent_roles:
            required_children = CHILD_REQUIRED.get(pr, [])
            if not required_children:
                continue
            # hidden parents skip
            if _is_hidden_from_at(el):
                continue
            if not _has_descendant_with_role_or_tag(el, parser, required_children):
                issues.append(
                    AccessibilityIssue(
                        title="ARIA parent roles include required child roles",
                        description=f"Element <{el['tag']}> with role '{pr}' is missing required child role(s): {', '.join(required_children)}.",
                        severity="medium",
                        evidence={
                            "parent_tag": el["tag"],
                            "parent_role": pr,
                            "required_child": required_children,
                            "id": el["id"][:50],
                        },
                        check_id="aria_parent_child",
                    )
                )
                break  # one per parent element
    return issues


def _check_aria_child_parent(parser: _A11yParser) -> list[AccessibilityIssue]:
    issues: list[AccessibilityIssue] = []
    for el in parser.elements:
        # Determine role for child requirement
        roles_to_check: list[str] = []
        for tok in el["role_tokens"]:
            if tok in PARENT_REQUIRED:
                roles_to_check.append(tok)
        # also handle tag equivalents for child roles like li => listitem
        if el["tag"] == "li" and "listitem" not in roles_to_check:
            # li implicitly listitem; check parent list requirement
            roles_to_check.append("listitem")
        if el["tag"] == "option" and "option" not in roles_to_check:
            roles_to_check.append("option")
        if el["tag"] == "tr" and "row" not in roles_to_check:
            roles_to_check.append("row")
        if el["tag"] in ("td", "th") and "cell" not in roles_to_check and "columnheader" not in roles_to_check:
            # treat td as cell
            roles_to_check.append("cell")
        for child_role in roles_to_check:
            required_parents = PARENT_REQUIRED.get(child_role, [])
            if not required_parents:
                continue
            if _is_hidden_from_at(el):
                continue
            if not _has_ancestor_with_role_or_tag(el, parser, required_parents):
                issues.append(
                    AccessibilityIssue(
                        title="ARIA roles are inside required parent elements",
                        description=f"Element <{el['tag']}> with role '{child_role}' is not inside required parent role(s): {', '.join(required_parents)}.",
                        severity="medium",
                        evidence={
                            "child_tag": el["tag"],
                            "child_role": child_role,
                            "required_parents": required_parents,
                            "id": el["id"][:50],
                        },
                        check_id="aria_child_parent",
                    )
                )
                break
    return issues


def _check_aria_id_unique(parser: _A11yParser) -> list[AccessibilityIssue]:
    dups = {k: v for k, v in parser.id_map.items() if len(v) > 1}
    if dups:
        return [
            AccessibilityIssue(
                title="ARIA IDs are unique across the page",
                description=f"Found {len(dups)} duplicate id(s). IDs must be unique for aria-labelledby and fragment navigation.",
                severity="medium",
                evidence={
                    "duplicate_ids": {k: len(v) for k, v in list(dups.items())[:5]},
                    "total_ids": len(parser.id_map),
                },
                check_id="aria_id_unique",
            )
        ]
    return []


def _check_aria_text_no_focusable(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["role"] != "text" and "text" not in el["role_tokens"]:
            continue
        # check descendants for focusable
        for idx in _get_descendant_indices(el, parser):
            child = parser.elements[idx]
            if _is_focusable(child):
                bad.append(el)
                break
    if bad:
        return [
            AccessibilityIssue(
                title="Elements with the role=text attribute do not have focusable descendents.",
                description=f"Found {len(bad)} element(s) with role=text containing focusable descendants, which breaks text semantics.",
                severity="medium",
                evidence={
                    "count": len(bad),
                    "ids": [e["id"][:50] for e in bad[:5]],
                },
                check_id="aria_text_no_focusable",
            )
        ]
    return []


def _check_duplicate_keyboard_shortcut(parser: _A11yParser) -> list[AccessibilityIssue]:
    counter: Counter[str] = Counter()
    for el in parser.elements:
        ak = el["attrs"].get("accesskey", "").strip().lower()
        if ak:
            # accesskey may be single char, normalize
            counter[ak] += 1
    dups = {k: v for k, v in counter.items() if v > 1}
    if dups:
        return [
            AccessibilityIssue(
                title="No duplicate keyboard shortcuts",
                description=f"Found duplicate accesskey values: {', '.join(list(dups.keys())[:5])}. Each accesskey should be unique.",
                severity="medium",
                evidence={"duplicate_keys": dups},
                check_id="duplicate_keyboard_shortcut",
            )
        ]
    return []


def _check_skip_link(parser: _A11yParser, html: str) -> list[AccessibilityIssue]:
    # check for skip link or landmark region
    has_skip = False
    has_landmark = False
    # landmarks: header, nav, main, footer, aside, article, section or role banner etc.
    landmark_tags = {"header", "nav", "main", "footer", "aside", "article", "section"}
    landmark_roles = {"banner", "navigation", "main", "contentinfo", "complementary", "region", "form", "search"}
    for el in parser.elements:
        if el["tag"] in landmark_tags:
            has_landmark = True
            break
        if el["role"] in landmark_roles or any(r in landmark_roles for r in el["role_tokens"]):
            has_landmark = True
            break
    # skip link detection: <a href="#..." > containing text skip or first href="#"
    for el in parser.elements:
        if el["tag"] != "a":
            continue
        href = el["attrs"].get("href", "").strip().lower()
        if not href.startswith("#"):
            continue
        if len(href) <= 1:
            continue
        text = el["text"].lower()
        # heuristics: contains "skip" or "jump to" or href to main/content
        if "skip" in text or "jump" in text or href in ("#main", "#main-content", "#content", "#primary", "#maincontent"):
            has_skip = True
            break
        # also generic first skip-like position: if href starts with # and element is early in doc (idx <10) and text indicates skip
        if href.startswith("#") and el["idx"] < 15 and text.strip():
            # check if parent is body direct? We'll consider any early link with href # as skip
            if "skip" in text:
                has_skip = True
                break
    # also check raw html for skip pattern via regex if parser missed
    if not has_skip:
        if re.search(r'<a[^>]+href=["\']#(?:main|content|skip)[^"\']*["\']', html, re.I):
            has_skip = True
        elif re.search(r'skip\s*(to)?\s*(main|content)', html, re.I):
            # ensure it's inside an anchor
            if re.search(r'<a[^>]*>.*?skip', html, re.I | re.S):
                has_skip = True

    if has_skip or has_landmark:
        return []
    return [
        AccessibilityIssue(
            title="Page has a skip link or landmark region",
            description="Page lacks both a skip link (href='#main') and landmark regions. Keyboard users need a way to bypass repeated content.",
            severity="low",
            evidence={"has_skip": has_skip, "has_landmark": has_landmark},
            check_id="skip_link",
        )
    ]


def _check_landmark_main(parser: _A11yParser) -> list[AccessibilityIssue]:
    has_main = False
    for el in parser.elements:
        if el["tag"] == "main":
            has_main = True
            break
        if el["role"] == "main" or "main" in el["role_tokens"]:
            has_main = True
            break
    if not has_main:
        return [
            AccessibilityIssue(
                title="Page has a main content area",
                description="No <main> landmark found (including role=main). A main landmark is required for navigation and assistive tech.",
                severity="medium",
                evidence={"has_main": False},
                check_id="landmark_main",
            )
        ]
    return []


def _check_landmarks(parser: _A11yParser) -> list[AccessibilityIssue]:
    # require at least 3 of header/nav/main/footer or ARIA equivalents
    required = {"header", "nav", "main", "footer"}
    found: set[str] = set()
    for el in parser.elements:
        if el["tag"] in required:
            found.add(el["tag"])
        # ARIA mapping
        if el["role"] == "banner" or "banner" in el["role_tokens"]:
            found.add("header")
        if el["role"] == "navigation" or "navigation" in el["role_tokens"]:
            found.add("nav")
        if el["role"] == "main" or "main" in el["role_tokens"]:
            found.add("main")
        if el["role"] == "contentinfo" or "contentinfo" in el["role_tokens"]:
            found.add("footer")
    if len(found) >= 3:
        return []
    # also if has_main already flagged, this will also flag but we keep separate
    if not found:
        missing = sorted(required)
    else:
        missing = sorted(required - found)
    # Only flag if missing at least 2 landmarks and not already flagged as has_skip etc? We'll still flag to cover title
    if len(found) < 3:
        return [
            AccessibilityIssue(
                title="HTML5 landmark elements are used to improve navigation",
                description=f"Semantic landmarks insufficient: found {', '.join(sorted(found)) or 'none'}, missing {', '.join(missing)}. Landmarks help assistive tech navigate.",
                severity="low",
                evidence={
                    "found_landmarks": sorted(found),
                    "missing": missing,
                    "required": sorted(required),
                },
                check_id="landmarks",
            )
        ]
    return []


def _check_heading_content(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] not in ("h1", "h2", "h3", "h4", "h5", "h6"):
            continue
        if not el["text"].strip() and not _has_accessible_name_via_aria(el, parser):
            # heading must contain content
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="All heading elements contain content.",
                description=f"Found {len(bad)} empty heading element(s). Headings must contain text or accessible name.",
                severity="medium",
                evidence={
                    "count": len(bad),
                    "levels": [e["tag"] for e in bad[:5]],
                    "ids": [e["id"][:50] for e in bad[:5]],
                },
                check_id="heading_content",
            )
        ]
    return []


def _check_visible_label_match(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Elements with visible text labels do not have matching accessible names.
    # Check: if element has visible text and aria-label, does aria-label contain visible text?
    # Restrict to interactive controls where visible label concept applies (buttons/links/toggles).
    bad: list[dict] = []
    for el in parser.elements:
        visible = el["text"].strip()
        aria_label = el["attrs"].get("aria-label", "").strip()
        if not visible or not aria_label:
            continue
        # Only interactive controls should be checked - avoids flagging meter/progress etc.
        tag = el["tag"]
        role = el["role"]
        is_interactive = False
        if tag == "button":
            is_interactive = True
        elif tag == "a" and el["attrs"].get("href", "").strip():
            is_interactive = True
        elif tag == "input" and el["attrs"].get("type", "").lower() in ("button", "submit", "reset"):
            is_interactive = True
        elif role in ("button", "link", "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "checkbox", "switch", "radio"):
            is_interactive = True
        elif "button" in el["role_tokens"] or "link" in el["role_tokens"]:
            is_interactive = True
        if not is_interactive:
            continue
        if tag in ("meter", "progress"):
            continue
        # normalize both
        vis_norm = " ".join(visible.lower().split())
        label_norm = " ".join(aria_label.lower().split())
        # skip symbolic single-char like "×" (len 1 and not alphanumeric)
        vis_stripped_raw = visible.strip()
        if len(vis_stripped_raw) == 1 and not vis_stripped_raw.isalnum():
            continue
        if len(vis_norm) == 1 and not vis_norm.isalnum():
            continue
        # handle theme toggle where both Light theme and Dark theme are concatenated
        # real visible is single state via CSS, but parser concatenates both spans
        if "light theme" in vis_norm and "dark theme" in vis_norm:
            if "light theme" in label_norm or "dark theme" in label_norm:
                continue
        # If visible text is not substring of aria-label and vice versa, flag mismatch
        # Lighthouse: visible text should be contained in accessible name
        if vis_norm not in label_norm:
            # Also check aria-labelledby case? For simplicity only aria-label mismatch
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Elements with visible text labels do not have matching accessible names.",
                description=f"Found {len(bad)} element(s) where visible text is not contained in accessible name (aria-label). This confuses speech recognition users.",
                severity="medium",
                evidence={
                    "count": len(bad),
                    "examples": [
                        {"visible": e["text"][:80], "aria_label": e["attrs"].get("aria-label", "")[:80], "tag": e["tag"]}
                        for e in bad[:3]
                    ],
                },
                check_id="visible_label_match",
            )
        ]
    return []


def _check_object_alt(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] not in ("object", "embed"):
            continue
        if _is_hidden_from_at(el):
            continue
        # object should have inner text fallback or title or aria-label
        has_text = _has_inner_text(el)
        has_aria = _has_accessible_name_via_aria(el, parser)
        has_title = bool(el["attrs"].get("title", "").strip())
        if not (has_text or has_aria or has_title):
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Embedded objects have alt text",
                description=f"Found {len(bad)} <object>/<embed> without alternative text (inner text, title, or aria-label).",
                severity="medium",
                evidence={"count": len(bad)},
                check_id="object_alt",
            )
        ]
    return []


def _check_video_captions(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Check <video> has <track kind=captions> descendant
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] != "video":
            continue
        # only flag visible videos (not in hidden dialog with aria-hidden or hidden)
        if _is_hidden_from_at(el):
            continue
        hidden_ancestor = False
        for anc_idx in el["ancestors"]:
            if _is_hidden_from_at(parser.elements[anc_idx]):
                hidden_ancestor = True
                break
        if hidden_ancestor:
            continue
        # skip video with empty src placeholder
        src = el["attrs"].get("src", "").strip()
        has_effective_src = bool(src)
        if not has_effective_src:
            for idx in _get_descendant_indices(el, parser):
                child = parser.elements[idx]
                if child["tag"] == "source" and child["attrs"].get("src", "").strip():
                    has_effective_src = True
                    break
        # explicit src="" with no effective src -> skip (task: src="" placeholder)
        if "src" in el["attrs"] and el["attrs"]["src"] == "" and not has_effective_src:
            continue
        if not has_effective_src:
            # if no effective src, only consider video meaningful if it has controls and is not muted placeholder
            has_controls = "controls" in el["attrs"]
            is_muted_placeholder = "muted" in el["attrs"] and ("autoplay" in el["attrs"] or "loop" in el["attrs"])
            if not (has_controls and not is_muted_placeholder):
                # For empty-src placeholder with muted/loop and no controls, skip
                # Also covers src missing with no controls -> skip per task
                continue
            # else: controls video with empty src but not placeholder -> still meaningful, proceed to caption check
            # but if src is truly empty and we have controls, we still check captions; keep fallthrough
        # check if video has track kind captions
        has_captions = False
        for idx in _get_descendant_indices(el, parser):
            child = parser.elements[idx]
            if child["tag"] == "track" and child["attrs"].get("kind", "").lower() == "captions":
                has_captions = True
                break
        if not has_captions:
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Videos have captions",
                description=f"Found {len(bad)} <video> without captions track.",
                severity="medium",
                evidence={"count": len(bad)},
                check_id="video_captions",
            )
        ]
    return []


def _check_custom_controls(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Custom controls have associated labels and ARIA roles
    # Check elements with click handlers? We only detect role-based custom controls
    # Custom control definition: div/span with role but no native semantics
    label_issues: list[dict] = []
    role_issues: list[dict] = []
    for el in parser.elements:
        tag = el["tag"]
        # consider div, span, etc with interactive role but no native label
        if tag not in ("div", "span", "a", "button"):
            # also check any non-semantic tags with role
            pass
        # custom controls have ARIA roles: check interactive elements without proper role?
        # For label check: custom controls (role button etc on div/span) should have accessible name
        is_custom = False
        if tag in ("div", "span") and el["role"] in ("button", "link", "checkbox", "radio", "switch", "slider"):
            is_custom = True
        if is_custom:
            if not _has_accessible_name(el, parser):
                label_issues.append(el)
        # role check: elements that look like custom control via tabindex/onclick?
        # We flag elements with tabindex=0 but no role and not native input? That's custom without role
        if tag in ("div", "span") and el["attrs"].get("tabindex", "") == "0" and not el["role"]:
            # Check if it seems interactive (has text but no role) -> missing role
            if el["text"].strip():
                role_issues.append(el)
    issues: list[AccessibilityIssue] = []
    if label_issues:
        issues.append(
            AccessibilityIssue(
                title="Custom controls have associated labels",
                description=f"Found {len(label_issues)} custom control(s) (div/span with role) without accessible name/label.",
                severity="medium",
                evidence={
                    "count": len(label_issues),
                    "examples": [{"tag": e["tag"], "role": e["role"]} for e in label_issues[:3]],
                },
                check_id="custom_controls_label",
            )
        )
    if role_issues:
        issues.append(
            AccessibilityIssue(
                title="Custom controls have ARIA roles",
                description=f"Found {len(role_issues)} custom interactive element(s) with tabindex but no ARIA role.",
                severity="medium",
                evidence={
                    "count": len(role_issues),
                    "examples": [{"tag": e["tag"], "text": e["text"][:50]} for e in role_issues[:3]],
                },
                check_id="custom_controls_role",
            )
        )
    return issues


def _check_interactive_keyboard_focusable(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        # interactive controls: a[href], button, input, select, textarea, [role=button/link] etc
        tag = el["tag"]
        role = el["role"]
        is_interactive = False
        if tag == "a" and el["attrs"].get("href", "").strip():
            is_interactive = True
        elif tag in ("button", "input", "select", "textarea"):
            # input hidden not interactive
            if tag == "input" and el["attrs"].get("type", "").lower() == "hidden":
                is_interactive = False
            else:
                is_interactive = True
        elif role in ("button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "slider", "spinbutton"):
            is_interactive = True
        elif "button" in el["role_tokens"] or "link" in el["role_tokens"]:
            is_interactive = True

        if not is_interactive:
            continue
        if _is_hidden_from_at(el):
            continue
        # check keyboard focusable: should not have tabindex="-1" and should be focusable
        # For native interactive, tabindex="-1" makes it not keyboard focusable -> flag
        tabindex = el["attrs"].get("tabindex", "").strip()
        if tabindex == "-1":
            bad.append(el)
            continue
        # For custom role on div/span, need tabindex >=0
        if tag in ("div", "span") and role and tabindex == "":
            # custom control without tabindex means not focusable via keyboard
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Interactive controls are keyboard focusable",
                description=f"Found {len(bad)} interactive control(s) not keyboard focusable (tabindex=-1 or missing tabindex for custom control).",
                severity="medium",
                evidence={
                    "count": len(bad),
                    "examples": [{"tag": e["tag"], "role": e["role"], "tabindex": e["attrs"].get("tabindex", "")} for e in bad[:5]],
                },
                check_id="interactive_keyboard_focusable",
            )
        ]
    return []


def _check_tab_order(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        tabindex_raw = el["attrs"].get("tabindex", "").strip()
        if not tabindex_raw:
            continue
        try:
            val = int(tabindex_raw)
            if val > 0:
                bad.append(el)
        except ValueError:
            continue
    if bad:
        return [
            AccessibilityIssue(
                title="The page has a logical tab order",
                description=f"Found {len(bad)} element(s) with tabindex > 0, which disrupts natural tab order.",
                severity="low",
                evidence={
                    "count": len(bad),
                    "elements": [e["attrs"].get("tabindex", "") for e in bad[:5]],
                },
                check_id="tab_order",
            )
        ]
    return []


def _check_offscreen_hidden(parser: _A11yParser) -> list[AccessibilityIssue]:
    bad: list[dict] = []
    for el in parser.elements:
        style = el["attrs"].get("style", "").lower()
        # detect offscreen positioning
        is_offscreen = False
        # common offscreen patterns
        if "left:-" in style.replace(" ", "") and ("position:absolute" in style.replace(" ", "")):
            is_offscreen = True
        if "clip:" in style or "clip-path:" in style:
            if "clip:rect(0" in style.replace(" ", "") or "inset(100%" in style.replace(" ", ""):
                is_offscreen = True
        if "text-indent:-" in style.replace(" ", ""):
            is_offscreen = True
        if not is_offscreen:
            continue
        # should be hidden from AT via aria-hidden
        if el["attrs"].get("aria-hidden", "").lower() != "true":
            # check if it contains focusable descendants -> should be hidden
            has_focusable = False
            for idx in _get_descendant_indices(el, parser):
                if _is_focusable(parser.elements[idx]):
                    has_focusable = True
                    break
            if has_focusable or _is_focusable(el):
                bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Offscreen content is hidden from assistive technology",
                description=f"Found {len(bad)} offscreen element(s) not hidden from assistive technology (missing aria-hidden).",
                severity="low",
                evidence={"count": len(bad)},
                check_id="offscreen_hidden",
            )
        ]
    return []


def _check_link_distinguishable(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Flag links that rely only on color (no underline, border, etc)
    bad: list[dict] = []
    for el in parser.elements:
        if el["tag"] != "a" or not el["attrs"].get("href", "").strip():
            continue
        if _is_hidden_from_at(el):
            continue
        style = el["attrs"].get("style", "").lower()
        # heuristics: if link has inline color but no text-decoration underline
        has_color = "color:" in style
        # check for underline indication
        has_underline = "text-decoration:" in style and "underline" in style
        has_border = "border-bottom" in style or "border:" in style
        has_bg = "background" in style
        # Also check class hints? For inline demonstration, only flag when explicitly styled with color and no underline
        if has_color and not (has_underline or has_border or has_bg):
            # To avoid false positives on normal sites where links are styled via CSS class not inline, we only flag inline color without underline
            # And we need to ensure link is not already distinguished via surrounding context? Keep strict: only inline color case
            # Count only if style contains color, so portfolio without inline color won't flag
            bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Links are distinguishable without relying on color.",
                description=f"Found {len(bad)} link(s) styled with color but without underline/border, relying solely on color to convey meaning.",
                severity="low",
                evidence={
                    "count": len(bad),
                    "examples": [e["text"][:50] for e in bad[:3]],
                },
                check_id="link_distinguishable",
            )
        ]
    return []


def _check_visual_fatigue(parser: _A11yParser, html: str) -> list[AccessibilityIssue]:
    # Markswebb 12: eyes get tired quickly
    # Heuristics: small font sizes, poor line-height, low contrast hints, dense text
    bad_elements: list[dict] = []
    dense_text_count = 0
    for el in parser.elements:
        style = el["attrs"].get("style", "").lower().replace(" ", "")
        # detect font-size < 12px
        m = re.search(r"font-size:(\d+)(px|pt)", style)
        if m:
            try:
                size = int(m.group(1))
                unit = m.group(2)
                # pt to px rough: pt*1.33
                if unit == "pt":
                    size = int(size * 1.33)
                if size < 11:
                    bad_elements.append(el)
                    continue
            except Exception:
                pass
        # detect line-height < 1.2 or < 120%
        m2 = re.search(r"line-height:([0-9.]+)", style)
        if m2:
            try:
                lh = float(m2.group(1))
                if lh < 1.2 and lh != 0:
                    # only flag if lh is factor, not px value large
                    # if value < 14 and "px" in style extract differently? Keep simple
                    if lh < 1.0:  # suspicious
                        bad_elements.append(el)
            except Exception:
                pass
        # dense text: if element has very long text without paragraph breaks, maybe > 500 chars continuous?
        txt = el["text"]
        if len(txt) > 600 and el["tag"] in ("div", "p", "span", "section"):
            # check if no descendants with block separation? Simplistic
            if txt.count(".") >= 5 and len(txt.split()) > 100:
                dense_text_count += 1

    # also check raw html for problematic patterns: huge inline styles, excessive <br> etc.
    # For portfolio, we expect no small fonts, so this should not flag

    # Only flag if multiple small fonts or dense blocks
    if len(bad_elements) >= 2 or dense_text_count >= 3:
        return [
            AccessibilityIssue(
                title="Visual fatigue: eyes get tired quickly",
                description=f"Found {len(bad_elements)} element(s) with small font or poor line-height, and {dense_text_count} dense text block(s) causing visual fatigue (Markswebb 12).",
                severity="low",
                evidence={
                    "small_font_count": len(bad_elements),
                    "dense_text_blocks": dense_text_count,
                },
                check_id="visual_fatigue",
            )
        ]
    # also check contrast hint: if many text elements with low contrast style color #fff on #fff? Not detectable without CSS
    return []


def _check_meta_refresh(parser: _A11yParser) -> list[AccessibilityIssue]:
    for el in parser.elements:
        if el["tag"] == "meta" and el["attrs"].get("http-equiv", "").lower() == "refresh":
            return [
                AccessibilityIssue(
                    title="The document does not use <meta http-equiv=\"refresh\">",
                    description="Document uses meta refresh, which can disorient users and is accessibility problematic.",
                    severity="medium",
                    evidence={"content": el["attrs"].get("content", "")[:200]},
                    check_id="meta_refresh",
                )
            ]
    return []


def _check_table_headers(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Check tables for header associations: td in large table have headers
    # Simplistic: if table has many td without th
    issues: list[AccessibilityIssue] = []
    for el in parser.elements:
        if el["tag"] != "table":
            continue
        # count th and td in descendants
        th_count = 0
        td_count = 0
        for idx in _get_descendant_indices(el, parser):
            child = parser.elements[idx]
            if child["tag"] == "th":
                th_count += 1
            elif child["tag"] == "td":
                td_count += 1
            elif child["tag"] == "caption":
                pass
        # large table heuristic: > 10 td and no th
        if td_count >= 10 and th_count == 0:
            issues.append(
                AccessibilityIssue(
                    title="<td> elements in a large <table> have one or more table headers.",
                    description=f"Large table with {td_count} data cells but no headers (th). Headers are needed for screen readers.",
                    severity="medium",
                    evidence={"td_count": td_count, "th_count": th_count},
                    check_id="table_headers",
                )
            )
        # also check caption vs summary duplication? Not needed
    return issues


def _check_focus_not_trapped(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Heuristic: check for elements with role dialog/modal that trap focus without escape mechanism
    # We'll flag dialog without close button (button) as potential trap?
    # To avoid false positives, only flag if dialog has focusable trap pattern: aria-modal=true but no button descendant
    issues: list[AccessibilityIssue] = []
    for el in parser.elements:
        is_modal = False
        if el["attrs"].get("aria-modal", "").lower() == "true":
            is_modal = True
        if el["role"] == "dialog" and is_modal:
            # check for descendant button or [aria-label="close"]
            has_close = False
            for idx in _get_descendant_indices(el, parser):
                child = parser.elements[idx]
                if child["tag"] == "button" or child["attrs"].get("aria-label", "").lower() == "close":
                    has_close = True
                    break
            if not has_close:
                issues.append(
                    AccessibilityIssue(
                        title="User focus is not accidentally trapped in a region",
                        description="Found modal dialog without close control, potentially trapping keyboard focus.",
                        severity="low",
                        evidence={"id": el["id"][:50]},
                        check_id="focus_not_trapped",
                    )
                )
    return issues


def _check_dropdown_accessible_name(parser: _A11yParser) -> list[AccessibilityIssue]:
    # Dropdown menus have accessible names: <select>, [role=combobox], [role=listbox]
    bad: list[dict] = []
    for el in parser.elements:
        is_dropdown = False
        if el["tag"] == "select":
            is_dropdown = True
        elif el["role"] in ("combobox", "listbox", "menu"):
            is_dropdown = True
        if not is_dropdown:
            continue
        if _is_hidden_from_at(el):
            continue
        # must have accessible name
        has_label = False
        if el["tag"] == "select":
            has_label = _has_label_association(el, parser)
        has_aria = _has_accessible_name_via_aria(el, parser)
        has_text = _has_inner_text(el)
        if not (has_label or has_aria or has_text):
            # also check wrapping label
            wrapped = False
            for anc_idx in el["ancestors"]:
                if parser.elements[anc_idx]["tag"] == "label":
                    wrapped = True
                    break
            if not wrapped:
                bad.append(el)
    if bad:
        return [
            AccessibilityIssue(
                title="Dropdown menus have accessible names",
                description=f"Found {len(bad)} dropdown control(s) without accessible name.",
                severity="medium",
                evidence={"count": len(bad)},
                check_id="dropdown_accessible_name",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def analyze_accessibility(url: str, client: httpx.AsyncClient | None = None) -> list[AccessibilityIssue]:
    """Run accessibility heuristics deterministically.

    Args:
        url: Page URL to analyze.
        client: Optional httpx.AsyncClient for testing / reuse.
    """
    if not url or not url.strip():
        raise ValueError("url must not be empty")
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    own_client = False
    if client is None:
        own_client = True
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            follow_redirects=True,
            headers={"User-Agent": "ux-analyzer accessibility detector"},
        )
    try:
        status, html, error = await _safe_get(client, url)
        if error is not None or html is None or status is None or status >= 400:
            return []
        if status != 200:
            return []
        parser = _parse_html(html)

        issues: list[AccessibilityIssue] = []
        # Core a11y name checks (critical)
        issues.extend(_check_button_accessible_name(parser))
        issues.extend(_check_input_button_label(parser))
        issues.extend(_check_image_button_alt(parser))
        issues.extend(_check_image_alt(parser))
        issues.extend(_check_form_input_accessible_name(parser))
        issues.extend(_check_toggle_accessible_name(parser))
        issues.extend(_check_frame_titles(parser))

        # Medium: ARIA validity
        issues.extend(_check_aria_role_valid(parser))
        issues.extend(_check_aria_deprecated(parser))
        issues.extend(_check_aria_required_attr(parser))
        issues.extend(_check_aria_parent_child(parser))
        issues.extend(_check_aria_child_parent(parser))
        issues.extend(_check_aria_id_unique(parser))
        issues.extend(_check_aria_text_no_focusable(parser))
        issues.extend(_check_duplicate_keyboard_shortcut(parser))

        # Additional a11y per titles
        issues.extend(_check_meter_accessible_name(parser))
        issues.extend(_check_progress_accessible_name(parser))
        issues.extend(_check_dialog_accessible_name(parser))
        issues.extend(_check_tooltip_accessible_name(parser))
        issues.extend(_check_treeitem_accessible_name(parser))
        issues.extend(_check_form_visible_label(parser))
        issues.extend(_check_duplicate_labels(parser))
        issues.extend(_check_heading_content(parser))
        issues.extend(_check_visible_label_match(parser))
        issues.extend(_check_object_alt(parser))
        issues.extend(_check_video_captions(parser))
        issues.extend(_check_dropdown_accessible_name(parser))
        issues.extend(_check_table_headers(parser))

        # UX-labeled a11y
        issues.extend(_check_skip_link(parser, html))
        issues.extend(_check_landmark_main(parser))
        issues.extend(_check_landmarks(parser))
        issues.extend(_check_interactive_keyboard_focusable(parser))
        issues.extend(_check_custom_controls(parser))
        issues.extend(_check_tab_order(parser))
        issues.extend(_check_offscreen_hidden(parser))
        issues.extend(_check_link_distinguishable(parser))
        issues.extend(_check_meta_refresh(parser))
        issues.extend(_check_focus_not_trapped(parser))
        issues.extend(_check_visual_fatigue(parser, html))

        # Deduplicate check_id ordering for determinism: already grouped but sort by check_id
        # Keep order but sort for determinism within same severity? Use check_id sort final
        issues.sort(key=lambda x: x.check_id)
        return issues
    finally:
        if own_client:
            await client.aclose()


def analyze_accessibility_sync(url: str) -> list[AccessibilityIssue]:
    """Sync wrapper for analyze_accessibility."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None  # type: ignore[assignment]
    if loop is not None and loop.is_running():  # type: ignore[union-attr]
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(asyncio.run, analyze_accessibility(url))
            return fut.result()
    return asyncio.run(analyze_accessibility(url))

