"""Imagery detector.

Covers Baymard #238 Quality of Photographs & Imagery and TheUXBites
imagery/performance overlapping issues.

Checks (at least 10, 12 implemented):

- hero_image_present (no_hero_imagery) - critical if no hero visual
- generic_stock (bespoke vs stock heuristic + filename stock) - medium
- low_resolution (small or missing dimensions, hero stricter) - critical if hero
- missing_alt_descriptive (informative images must have alt) - medium
- generic_alt (alt = "image"/"photo" etc) - medium
- lazy_lcp (largest image lazy-loaded above-fold) - medium
- preload_lcp (LCP image missing preload) - medium
- modern_format (outdated jpg/png without webp/avif or picture) - low
- cutout_vs_lifestyle (hero cut-out on white vs lifestyle) - low
- image_count_balance (too few / too many vs content) - low
- unoptimized_loading (below-fold missing lazy / decoding async) - low
- missing_dimensions detail (part of low_resolution but explicit evidence) - low (merged)

Imagery slice uses static httpx fetch + HTMLParser (no browser required).
Evidence includes counts, srcs, dimensions, alt values.

Severity:
- critical: no hero, low-res hero
- medium: stock generic, missing alt descriptive, generic alt, lazy LCP, preload LCP
- low: format, cut-out vs lifestyle, image count, unoptimized loading
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True, slots=True)
class ImageryIssue:
    """Evidence-backed imagery finding."""

    title: str
    description: str
    severity: str  # critical|medium|low
    evidence: dict
    check_id: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOCK_KEYWORDS = [
    "unsplash",
    "pexels",
    "shutterstock",
    "istock",
    "getty",
    "stock",
    "placeholder",
    "picsum",
    "lorem",
    "dummy",
    "filler",
    "placehold",
    "via.placeholder",
    "placeimg",
    "fakeimg",
]

CUTOUT_KEYWORDS = [
    "cutout",
    "cut-out",
    "cut_out",
    "isolated",
    "white-bg",
    "white_bg",
    "transparent-bg",
    "transparent_bg",
    "on-white",
    "ghost-mannequin",
    "clipping",
]

GENERIC_ALTS = {
    "image",
    "photo",
    "picture",
    "pic",
    "img",
    "photo image",
    "placeholder image",
    "stock photo",
    "stock image",
    "dummy image",
    "placeholder",
    "thumbnail",
    "banner",
    "hero image",
    "test image",
    "image 1",
    "photo 1",
    "picture 1",
}

HERO_KEYWORDS = ["hero", "banner", "jumbotron", "masthead", "cover", "billboard"]

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
# Parser
# ---------------------------------------------------------------------------
class _ImageryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[dict] = []
        self.canvases: list[dict] = []
        self.videos: list[dict] = []
        self.svgs: list[dict] = []
        self.sources: list[dict] = []
        self.links: list[dict] = []
        self.stack: list[dict] = []  # {tag, is_hero}
        self.hero_depth: int = 0
        self.in_head: bool = False
        self.in_style: bool = False
        self.in_script: bool = False
        self.inside_picture: bool = False
        self.visible_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        d = {k.lower(): (v or "") for k, v in attrs}

        if t == "head":
            self.in_head = True
        if t == "style":
            self.in_style = True
        if t == "script":
            self.in_script = True
        if t == "picture":
            self.inside_picture = True
        if t == "source":
            self.sources.append(d)
        if t == "link":
            self.links.append(d)

        # hero container detection
        is_hero_container = False
        cls = d.get("class", "").lower()
        ident = d.get("id", "").lower()
        for kw in HERO_KEYWORDS:
            if kw in cls or kw in ident:
                is_hero_container = True
                break

        # push non-void
        if t not in VOID_ELEMENTS:
            self.stack.append({"tag": t, "is_hero": is_hero_container})
            if is_hero_container:
                self.hero_depth += 1

        if t == "img":
            rec = {
                "src": d.get("src", "").strip(),
                "srcset": d.get("srcset", "").strip(),
                "sizes": d.get("sizes", "").strip(),
                "width": d.get("width", "").strip(),
                "height": d.get("height", "").strip(),
                "alt": d.get("alt", "") if "alt" in d else None,  # None means missing
                "attrs": d,
                "loading": d.get("loading", "").strip(),
                "decoding": d.get("decoding", "").strip(),
                "fetchpriority": d.get("fetchpriority", "").strip(),
                "class": d.get("class", ""),
                "id": d.get("id", ""),
                "in_hero": self.hero_depth > 0,
                "order": len(self.images),
                "inside_picture": self.inside_picture,
            }
            self.images.append(rec)
        elif t == "canvas":
            rec = {
                "class": d.get("class", ""),
                "id": d.get("id", ""),
                "attrs": d,
                "in_hero": self.hero_depth > 0,
                "decorative": d.get("aria-hidden", "").lower() == "true",
            }
            self.canvases.append(rec)
        elif t == "video":
            rec = {
                "src": d.get("src", "").strip() or d.get("poster", "").strip(),
                "poster": d.get("poster", "").strip(),
                "class": d.get("class", ""),
                "attrs": d,
                "in_hero": self.hero_depth > 0,
            }
            self.videos.append(rec)
        elif t == "svg":
            cls_lower = d.get("class", "").lower()
            aria_hidden = d.get("aria-hidden", "").lower() == "true"
            parent_is_button = any(entry.get("tag") == "button" for entry in self.stack)
            # broader decorative detection: icons inside buttons, toggle icons, or generic 24px icons without class
            view_box = d.get("viewBox", "") or d.get("viewbox", "")
            decorative = (
                aria_hidden
                or "icon" in cls_lower
                or "toggle" in cls_lower
                or parent_is_button
                or (not cls_lower and view_box == "0 0 24 24")
            )
            rec = {
                "class": d.get("class", ""),
                "attrs": d,
                "in_hero": self.hero_depth > 0,
                "decorative": decorative,
            }
            self.svgs.append(rec)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "head":
            self.in_head = False
        elif t == "style":
            self.in_style = False
        elif t == "script":
            self.in_script = False
        elif t == "picture":
            self.inside_picture = False

        if self.stack and self.stack[-1]["tag"] == t:
            popped = self.stack.pop()
            if popped["is_hero"]:
                self.hero_depth -= 1
                if self.hero_depth < 0:
                    self.hero_depth = 0
        else:
            # search from top for matching tag (malformed HTML)
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i]["tag"] == t:
                    while len(self.stack) > i:
                        popped = self.stack.pop()
                        if popped["is_hero"]:
                            self.hero_depth -= 1
                            if self.hero_depth < 0:
                                self.hero_depth = 0
                    break

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # self-closing like <img/> or <source/>
        self.handle_starttag(tag, attrs)
        t = tag.lower()
        if t not in VOID_ELEMENTS and self.stack and self.stack[-1]["tag"] == t:
            popped = self.stack.pop()
            if popped["is_hero"]:
                self.hero_depth -= 1
                if self.hero_depth < 0:
                    self.hero_depth = 0

    def handle_data(self, data: str) -> None:
        if self.in_style or self.in_script or self.in_head:
            return
        txt = data.strip()
        if not txt:
            return
        norm = " ".join(txt.split())
        if norm:
            self.visible_text_parts.append(norm)

    @property
    def visible_text(self) -> str:
        return " ".join(self.visible_text_parts)


def _parse_html(html: str) -> _ImageryParser:
    p = _ImageryParser()
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


def _parse_int(s: str) -> int | None:
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _parse_dimensions(img: dict) -> tuple[int | None, int | None]:
    w = _parse_int(img.get("width", ""))
    h = _parse_int(img.get("height", ""))
    return w, h


def _is_stock(img: dict) -> bool:
    src = (img.get("src") or "").lower()
    alt = (img.get("alt") or "").lower() if img.get("alt") is not None else ""
    cls = (img.get("class") or "").lower()
    combined = f"{src} {alt} {cls}"
    # also check filename
    filename = src.split("/")[-1].split("?")[0].lower()
    for kw in STOCK_KEYWORDS:
        if kw in src or kw in filename or kw in combined:
            return True
    return False


def _is_generic_alt(alt: str | None) -> bool:
    if alt is None:
        return False
    norm = " ".join(alt.strip().lower().split())
    if not norm:
        return False
    if norm in GENERIC_ALTS:
        return True
    if re.match(r"^(image|photo|picture|pic|img)\s*\d*$", norm):
        return True
    # single generic word
    if norm in {"image", "photo", "picture", "pic", "img"}:
        return True
    return False


def _find_lcp_image(images: list[dict]) -> dict | None:
    if not images:
        return None
    best: dict | None = None
    best_area = -1
    best_idx = 10**9
    for idx, img in enumerate(images):
        src = img.get("src", "").strip()
        if not src or src.startswith("data:"):
            continue
        w, h = _parse_dimensions(img)
        area = (w * h) if w and h else 0
        # Prefer images with area; if tie, earlier wins (above-fold)
        # Also prefer non-svg for LCP if possible? But include all
        if area > best_area or (area == best_area and idx < best_idx):
            best = img
            best_area = area
            best_idx = idx
    # Fallback to first non-data image if all area 0
    if best is None:
        for img in images:
            src = img.get("src", "").strip()
            if src and not src.startswith("data:"):
                return img
    return best


def _has_modern_source(parser: _ImageryParser) -> bool:
    for s in parser.sources:
        typ = s.get("type", "").lower()
        srcset = s.get("srcset", "").lower()
        src = s.get("src", "").lower()
        if typ == "image/webp" or typ == "image/avif":
            return True
        if ".webp" in srcset or ".avif" in srcset or ".webp" in src or ".avif" in src:
            return True
    return False


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def _check_no_hero(parser: _ImageryParser) -> ImageryIssue | None:
    hero_imgs = [i for i in parser.images if i["in_hero"]]
    hero_canvases = [c for c in parser.canvases if c["in_hero"]]
    hero_videos = [v for v in parser.videos if v["in_hero"] and (v.get("src", "").strip() or v.get("poster", "").strip())]
    hero_svgs = [s for s in parser.svgs if s["in_hero"] and not s.get("decorative")]

    total_hero_visuals = len(hero_imgs) + len(hero_canvases) + len(hero_videos) + len(hero_svgs)

    if total_hero_visuals > 0:
        return None

    # Fallback: large image near top considered hero even without hero class
    # If first 2 images have width >=600, treat as hero present for sites without hero class
    for img in parser.images[:2]:
        w, _ = _parse_dimensions(img)
        if w is not None and w >= 600:
            return None
        # also check if image likely lifestyle large banner via width attribute missing but fetchpriority high
        if img.get("fetchpriority", "").lower() == "high":
            return None

    # If page is very short (little content), hero not required? But still flag for completeness
    # Only flag if page has visible content > 200 chars (avoid flagging empty/error pages already handled)
    if len(parser.visible_text) < 100:
        return None

    evidence = {
        "total_images": len(parser.images),
        "hero_images": len(hero_imgs),
        "hero_canvases": len(hero_canvases),
        "hero_videos": len(hero_videos),
        "hero_svgs": len(hero_svgs),
        "image_srcs": [i.get("src", "")[:200] for i in parser.images[:5]],
        "canvas_classes": [c.get("class", "")[:100] for c in parser.canvases[:3]],
    }
    return ImageryIssue(
        title="No hero/bespoke imagery - page lacks prominent visual",
        description=(
            "No prominent hero visual found in above-fold (no image/canvas/video inside hero/banner section "
            "and no large top image). High-quality bespoke hero imagery entices users and communicates brand feel."
        ),
        severity="critical",
        evidence=evidence,
        check_id="no_hero_imagery",
    )


def _check_generic_stock(parser: _ImageryParser) -> ImageryIssue | None:
    flagged = [i for i in parser.images if _is_stock(i)]
    if not flagged:
        return None
    evidence = {
        "count": len(flagged),
        "srcs": [f.get("src", "")[:300] for f in flagged[:5]],
        "alts": [f.get("alt", "")[:100] for f in flagged[:5]],
        "total_images": len(parser.images),
    }
    return ImageryIssue(
        title="Generic stock imagery detected",
        description=(
            f"Found {len(flagged)} image(s) with stock/placeholder signals (unsplash, stock, picsum, placeholder, etc.). "
            "Generic stock alienates users; bespoke photography entices."
        ),
        severity="medium",
        evidence=evidence,
        check_id="generic_stock",
    )


def _check_low_resolution(parser: _ImageryParser) -> ImageryIssue | None:
    flagged: list[dict] = []
    hero_flagged: list[dict] = []
    missing_dims: list[dict] = []
    for img in parser.images:
        src = (img.get("src") or "").lower()
        if not src or src.startswith("data:"):
            continue
        if src.endswith(".svg"):
            continue  # vector is resolution-independent
        cls = (img.get("class") or "").lower()
        if "icon" in cls or "logo" in cls or "favicon" in cls:
            continue
        w, h = _parse_dimensions(img)
        in_hero = img["in_hero"]
        if w is None or h is None:
            missing_dims.append(img)
            flagged.append(img)
            if in_hero:
                hero_flagged.append(img)
            continue
        if in_hero:
            if w < 400 or h < 200 or (w * h) < 80000:
                flagged.append(img)
                hero_flagged.append(img)
        else:
            if w < 200 or h < 200:
                flagged.append(img)

    if not flagged:
        return None

    severity = "critical" if hero_flagged else "medium"
    title = "Low-resolution or un-sized hero imagery" if hero_flagged else "Low-resolution imagery"
    evidence = {
        "count": len(flagged),
        "hero_flagged": len(hero_flagged),
        "missing_dimensions": len(missing_dims),
        "details": [
            {
                "src": f.get("src", "")[:300],
                "width": f.get("width", ""),
                "height": f.get("height", ""),
                "in_hero": f["in_hero"],
            }
            for f in flagged[:5]
        ],
        "total_images": len(parser.images),
    }
    description = (
        "Low-resolution or un-sized images turn off users and feel fake. "
        f"Flagged {len(flagged)} image(s) ({len(missing_dims)} missing dimensions, {len(hero_flagged)} in hero)."
    )
    return ImageryIssue(
        title=title,
        description=description,
        severity=severity,
        evidence=evidence,
        check_id="low_resolution",
    )


def _check_missing_dimensions(parser: _ImageryParser) -> ImageryIssue | None:
    # Separate explicit check for missing width/height to give clear evidence (distinct from low_res)
    missing = []
    for img in parser.images:
        src = (img.get("src") or "").strip()
        if not src or src.startswith("data:"):
            continue
        if src.lower().endswith(".svg"):
            continue
        w, h = _parse_dimensions(img)
        if w is None or h is None:
            missing.append(img)
    if not missing:
        return None
    evidence = {
        "count": len(missing),
        "total_images": len(parser.images),
        "missing_srcs": [m.get("src", "")[:200] for m in missing[:5]],
        "examples": [
            {"src": m.get("src", "")[:200], "width": m.get("width", ""), "height": m.get("height", "")}
            for m in missing[:5]
        ],
    }
    return ImageryIssue(
        title="Images missing explicit dimensions",
        description="Images without width/height cause layout shifts, delay visual stability, and cannot guarantee high-quality rendering.",
        severity="low",
        evidence=evidence,
        check_id="missing_dimensions",
    )


def _check_missing_alt(parser: _ImageryParser) -> ImageryIssue | None:
    flagged = []
    for img in parser.images:
        attrs = img["attrs"]
        alt = img.get("alt")
        if alt is None:
            # Missing alt attribute entirely -> informative image without description
            # Exclude decorative if aria-hidden or role presentation? But missing alt without decorative hint is still flag
            if attrs.get("aria-hidden", "").lower() == "true" or attrs.get("role", "").lower() in ("presentation", "none"):
                continue
            flagged.append(img)
            continue
        # alt present but empty string -> check decorative
        if isinstance(alt, str) and alt.strip() == "":
            decorative = False
            if attrs.get("aria-hidden", "").lower() == "true" or attrs.get("role", "").lower() in ("presentation", "none"):
                decorative = True
            cls = (attrs.get("class") or "").lower()
            if "decorative" in cls or "icon" in cls:
                decorative = True
            if not decorative:
                flagged.append(img)
    if not flagged:
        return None
    evidence = {
        "count": len(flagged),
        "total_images": len(parser.images),
        "srcs": [f.get("src", "")[:200] for f in flagged[:5]],
        "alts": [f.get("alt", "") if f.get("alt") is not None else "<missing>" for f in flagged[:5]],
    }
    return ImageryIssue(
        title="Informative images missing descriptive alt",
        description=(
            f"Found {len(flagged)} informative image(s) without descriptive alt text. "
            "Informative images must have descriptive alt; decorative images should have alt=\"\" with aria-hidden or presentation role."
        ),
        severity="medium",
        evidence=evidence,
        check_id="missing_alt",
    )


def _check_generic_alt(parser: _ImageryParser) -> ImageryIssue | None:
    flagged = [i for i in parser.images if i.get("alt") is not None and _is_generic_alt(i.get("alt", ""))]
    if not flagged:
        return None
    evidence = {
        "count": len(flagged),
        "alts": [f.get("alt", "")[:100] for f in flagged[:5]],
        "srcs": [f.get("src", "")[:200] for f in flagged[:5]],
        "total_images": len(parser.images),
    }
    return ImageryIssue(
        title='Generic alt text ("image", "photo", etc.)',
        description=(
            f"Found {len(flagged)} image(s) with generic alt like 'image'/'photo' instead of descriptive text. "
            "Generic alt fails to convey image purpose and feels like placeholder stock."
        ),
        severity="medium",
        evidence=evidence,
        check_id="generic_alt",
    )


def _check_lazy_lcp(parser: _ImageryParser) -> ImageryIssue | None:
    lcp = _find_lcp_image(parser.images)
    if lcp is None:
        return None
    loading = (lcp.get("loading") or "").lower()
    if loading == "lazy":
        evidence = {
            "lcp_src": lcp.get("src", "")[:300],
            "loading": loading,
            "width": lcp.get("width", ""),
            "height": lcp.get("height", ""),
            "in_hero": lcp.get("in_hero"),
            "fetchpriority": lcp.get("fetchpriority", ""),
        }
        return ImageryIssue(
            title="Largest image is unnecessarily lazy-loaded",
            description="The likely Largest Contentful Paint image has loading=\"lazy\", which defers above-fold hero imagery and hurts perceived performance.",
            severity="medium",
            evidence=evidence,
            check_id="lazy_lcp",
        )
    return None


def _check_preload_lcp(parser: _ImageryParser) -> ImageryIssue | None:
    """Imagery perspective LCP preload: largest image should be preloaded for performance."""
    lcp = _find_lcp_image(parser.images)
    if lcp is None:
        return None
    src = (lcp.get("src") or "").strip()
    if not src or src.lower().startswith("data:"):
        return None
    # Vector SVGs don't require preload; skip
    if src.lower().endswith(".svg"):
        return None
    # If LCP is lazy or small icon, skip preload requirement?
    cls = (lcp.get("class") or "").lower()
    if "icon" in cls or "logo" in cls:
        return None
    w, h = _parse_dimensions(lcp)
    area = (w * h) if w and h else 0
    # If no dimensions and not in hero, likely not LCP candidate for preload (avoid false positive on tiny thumbs)
    # Still flag if in hero or large.
    if area == 0 and not lcp.get("in_hero"):
        # Fallback: consider non-hero tiny images not requiring preload
        # But if fetchpriority high or width>=600, still require
        if lcp.get("fetchpriority", "").lower() != "high":
            # check width fallback
            if w is None or w < 400:
                return None
    for link in parser.links:
        rel = (link.get("rel") or "").lower()
        if "preload" not in rel:
            continue
        href = (link.get("href") or "").strip()
        if not href:
            continue
        as_attr = (link.get("as") or "").lower()
        if as_attr and as_attr != "image":
            continue
        src_file = src.split("/")[-1].split("?")[0].lower()
        href_file = href.split("/")[-1].split("?")[0].lower()
        if href_file and src_file and href_file == src_file:
            return None
        if href_file and src_file and href_file in src_file or src_file in href_file:
            return None
        low_href = href.lower()
        low_src = src.lower()
        if low_href in low_src or low_src in low_href:
            return None
        imagesrcset = (link.get("imagesrcset") or "").lower()
        if src_file and src_file in imagesrcset:
            return None
        # also check image srcset via link's href matching srcset?
        # direct href exact match
        if href == src:
            return None
    evidence = {
        "lcp_src": src[:300],
        "width": lcp.get("width", ""),
        "height": lcp.get("height", ""),
        "in_hero": lcp.get("in_hero"),
        "fetchpriority": lcp.get("fetchpriority", ""),
        "has_preload": False,
        "total_images": len(parser.images),
        "suggestion": 'Add <link rel="preload" as="image" href="..." fetchpriority="high"> for LCP',
    }
    return ImageryIssue(
        title="LCP image missing preload",
        description="The likely Largest Contentful Paint image is not preloaded. Preload LCP with <link rel=preload as=image> and fetchpriority high to improve perceived performance (TheUXBites).",
        severity="medium",
        evidence=evidence,
        check_id="preload_lcp",
    )


def _check_modern_format(parser: _ImageryParser) -> ImageryIssue | None:
    if not parser.images:
        return None
    has_modern_global = _has_modern_source(parser)
    # Check if any image is modern via src or srcset
    modern_found = has_modern_global
    outdated: list[dict] = []
    for img in parser.images:
        src = (img.get("src") or "").lower()
        if not src or src.startswith("data:"):
            continue
        if src.endswith(".svg"):
            modern_found = True
            continue
        if src.endswith(".webp") or src.endswith(".avif"):
            modern_found = True
            continue
        srcset = (img.get("srcset") or "").lower()
        if ".webp" in srcset or ".avif" in srcset:
            modern_found = True
            continue
        if src.endswith(".jpg") or src.endswith(".jpeg") or src.endswith(".png"):
            # Check if it's an icon/logo -> ignore format for tiny icons?
            cls = (img.get("class") or "").lower()
            if "icon" in cls or "logo" in cls:
                continue
            outdated.append(img)
        elif src.endswith(".gif"):
            outdated.append(img)

    if outdated and not modern_found:
        evidence = {
            "outdated_count": len(outdated),
            "total_images": len(parser.images),
            "srcs": [o.get("src", "")[:200] for o in outdated[:5]],
            "has_modern_alternative": modern_found,
            "suggestion": "Serve WebP/AVIF via <picture> or srcset, keep jpg/png as fallback",
        }
        return ImageryIssue(
            title="Outdated image format (no WebP/AVIF)",
            description=(
                f"All {len(outdated)} image(s) use outdated formats (jpg/png/gif) without WebP/AVIF alternative. "
                "Modern formats reduce bytes and preserve quality."
            ),
            severity="low",
            evidence=evidence,
            check_id="modern_format",
        )
    return None


def _check_cutout(parser: _ImageryParser) -> ImageryIssue | None:
    flagged = []
    for img in parser.images:
        if not img.get("in_hero"):
            continue
        src = (img.get("src") or "").lower()
        cls = (img.get("class") or "").lower()
        alt = (img.get("alt") or "").lower() if img.get("alt") is not None else ""
        combined = f"{src} {cls} {alt}"
        for kw in CUTOUT_KEYWORDS:
            if kw in combined:
                flagged.append(img)
                break
    if not flagged:
        return None
    evidence = {
        "count": len(flagged),
        "srcs": [f.get("src", "")[:300] for f in flagged[:3]],
        "hero_images": len([i for i in parser.images if i["in_hero"]]),
    }
    return ImageryIssue(
        title="Cut-out vs lifestyle mismatch",
        description=(
            "Hero imagery appears to be isolated cut-out on white/transparent rather than contextual lifestyle. "
            "On homepages cut-outs are less effective at conveying brand feel than lifestyle scenes."
        ),
        severity="low",
        evidence=evidence,
        check_id="cutout_vs_lifestyle",
    )


def _check_image_count(parser: _ImageryParser) -> ImageryIssue | None:
    total_imgs = len([i for i in parser.images if (i.get("src", "").strip() and not i.get("src", "").strip().startswith("data:"))])
    # total visuals includes canvas/video non-decorative + images
    total_canvases = len([c for c in parser.canvases if not c.get("decorative")])
    total_videos = len([v for v in parser.videos if (v.get("src", "").strip() or v.get("poster", "").strip())])
    total_svgs_content = len([s for s in parser.svgs if not s.get("decorative")])
    total_visuals = total_imgs + total_canvases + total_videos + total_svgs_content

    text = parser.visible_text
    word_count = len(text.split()) if text else 0
    text_len = len(text)

    # Too few: wordy page with very few visuals
    if word_count > 300 and total_visuals < 2:
        evidence = {
            "total_images": total_imgs,
            "total_canvases": total_canvases,
            "total_videos": total_videos,
            "total_svgs_content": total_svgs_content,
            "total_visuals": total_visuals,
            "word_count": word_count,
            "text_len": text_len,
            "srcs": [i.get("src", "")[:200] for i in parser.images[:3]],
        }
        return ImageryIssue(
            title="Too few images for content length",
            description=(
                f"Page has {word_count} words but only {total_visuals} visual(s) ({total_imgs} <img>). "
                "No pictures at all, or too few, fails to give users a glimpse and reduces attention."
            ),
            severity="low",
            evidence=evidence,
            check_id="image_count",
        )

    # Too many heavy images
    if total_imgs > 15:
        # Check if many images lack lazy optimization (to distinguish intentional gallery vs fatigue)
        heavy = total_imgs
        evidence = {
            "total_images": total_imgs,
            "total_visuals": total_visuals,
            "word_count": word_count,
            "heavy_count": heavy,
        }
        return ImageryIssue(
            title="Too many heavy images (potential fatigue)",
            description=(
                f"Page has {total_imgs} images, which may overwhelm and tire eyes. "
                "Consider curating inspirational product images and lazy-loading below-fold."
            ),
            severity="low",
            evidence=evidence,
            check_id="image_count",
        )

    return None


def _check_unoptimized_loading(parser: _ImageryParser) -> ImageryIssue | None:
    if len(parser.images) < 2:
        return None
    flagged: list[dict] = []
    for idx, img in enumerate(parser.images):
        src = (img.get("src") or "").lower()
        if not src or src.startswith("data:") or src.endswith(".svg"):
            continue
        loading = (img.get("loading") or "").lower()
        # below-fold heuristic: index >=1 (first image is LCP) or not in hero
        is_below = idx >= 1 or not img.get("in_hero")
        if is_below and loading != "lazy":
            # Check if image is not icon
            cls = (img.get("class") or "").lower()
            if "icon" in cls or "logo" in cls:
                continue
            flagged.append(img)

    if not flagged:
        return None

    # Also check decoding async missing for at least half?
    missing_decoding = []
    for img in parser.images:
        src = (img.get("src") or "").lower()
        if not src or src.startswith("data:") or src.endswith(".svg"):
            continue
        dec = (img.get("decoding") or "").lower()
        if dec != "async":
            missing_decoding.append(img)

    evidence = {
        "total_images": len(parser.images),
        "below_fold_missing_lazy": len(flagged),
        "missing_decoding_async": len(missing_decoding),
        "flagged_srcs": [f.get("src", "")[:200] for f in flagged[:5]],
        "example_decoding_missing": [m.get("src", "")[:200] for m in missing_decoding[:3]],
    }
    # provide suggestion
    return ImageryIssue(
        title="Unoptimized image loading (missing lazy/decoding)",
        description=(
            f"Found {len(flagged)} below-fold image(s) without loading=\"lazy\" "
            f"and {len(missing_decoding)} without decoding=\"async\". "
            "Below-fold images should be lazy-loaded; decoding async avoids blocking."
        ),
        severity="low",
        evidence=evidence,
        check_id="unoptimized_loading",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def analyze_imagery(url: str, client: httpx.AsyncClient | None = None) -> list[ImageryIssue]:
    """Run imagery heuristics deterministically.

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
            headers={"User-Agent": "ux-analyzer imagery detector"},
        )
    try:
        status, html, error = await _safe_get(client, url)
        if error is not None or html is None or status is None or status >= 400:
            return []
        if status != 200:
            return []
        parser = _parse_html(html)

        issues: list[ImageryIssue] = []

        for fn in [
            _check_no_hero,
            _check_generic_stock,
            _check_low_resolution,
            _check_missing_dimensions,
            _check_missing_alt,
            _check_generic_alt,
            _check_lazy_lcp,
            _check_preload_lcp,
            _check_modern_format,
            _check_cutout,
            _check_image_count,
            _check_unoptimized_loading,
        ]:
            try:
                res = fn(parser)
            except Exception:
                continue
            if res is not None:
                # _check calls return single issue; but some could return list in future
                if isinstance(res, list):
                    issues.extend(res)
                else:
                    issues.append(res)

        issues.sort(key=lambda x: x.check_id)
        return issues
    finally:
        if own_client:
            await client.aclose()


def analyze_imagery_sync(url: str) -> list[ImageryIssue]:
    """Sync wrapper for analyze_imagery."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None  # type: ignore[assignment]
    if loop is not None and loop.is_running():  # type: ignore[union-attr]
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(asyncio.run, analyze_imagery(url))
            return fut.result()
    return asyncio.run(analyze_imagery(url))
