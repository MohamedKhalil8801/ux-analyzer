"""Color parsing + classification helpers.

Port of @slop-detect/core `color.ts` (MIT) with a Python-side CSS color
parser standing in for the browser canvas-based `parseColor`. Computed styles
from the snapshot are serialized as `rgb()/rgba()` or hex, which this parser
covers along with the named colors a canvas context would accept.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Color:
    r: float
    g: float
    b: float
    a: float = 1.0
    approx: bool = False


_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3,8})$")
_RGB_FN_RE = re.compile(
    r"rgba?\(\s*([\d.]+%?)\s*[, ]\s*([\d.]+%?)\s*[, ]\s*([\d.]+%?)(?:\s*[,/]\s*([\d.]+%?))?\s*\)"
)
_HSL_FN_RE = re.compile(
    r"hsla?\(\s*([\d.]+)(?:deg)?\s*[, ]\s*([\d.]+)%\s*[, ]\s*([\d.]+)%(?:\s*[,/]\s*([\d.]+%?))?\s*\)"
)

_NAMED: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "violet": (238, 130, 238),
    "magenta": (255, 0, 255),
    "fuchsia": (255, 0, 255),
    "cyan": (0, 255, 255),
    "aqua": (0, 255, 255),
    "teal": (0, 128, 128),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "lime": (0, 255, 0),
    "navy": (0, 0, 128),
    "indigo": (75, 0, 130),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gold": (255, 215, 0),
    "beige": (245, 245, 220),
    "coral": (255, 127, 80),
    "crimson": (220, 20, 60),
    "salmon": (250, 128, 114),
    "tomato": (255, 99, 71),
    "khaki": (240, 230, 140),
    "ivory": (255, 255, 240),
    "azure": (240, 255, 255),
    "lavender": (230, 230, 250),
    "mintcream": (245, 255, 250),
    "linen": (250, 240, 230),
    "snow": (255, 250, 250),
    "gainsboro": (220, 220, 220),
    "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211),
    "darkgray": (169, 169, 169),
    "darkgrey": (169, 169, 169),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "whitesmoke": (245, 245, 245),
    "aliceblue": (240, 248, 255),
    "lightcyan": (224, 255, 255),
    "paleturquoise": (175, 238, 238),
    "mediumpurple": (147, 112, 219),
    "rebeccapurple": (102, 51, 153),
    "slateblue": (106, 90, 205),
    "darkslateblue": (72, 61, 139),
    "royalblue": (65, 105, 225),
    "steelblue": (70, 130, 180),
    "skyblue": (135, 206, 235),
    "lightblue": (173, 216, 230),
    "darkblue": (0, 0, 139),
    "mediumblue": (0, 0, 205),
    "midnightblue": (25, 25, 112),
    "darkgreen": (0, 100, 0),
    "forestgreen": (34, 139, 34),
    "seagreen": (46, 139, 87),
    "limegreen": (50, 205, 50),
    "lightgreen": (144, 238, 144),
    "darkred": (139, 0, 0),
    "firebrick": (178, 34, 34),
    "darkorange": (255, 140, 0),
    "goldenrod": (218, 165, 32),
    "darkkhaki": (189, 183, 107),
    "tan": (210, 180, 140),
    "chocolate": (210, 105, 30),
    "saddlebrown": (139, 69, 19),
    "cornsilk": (255, 248, 220),
    "oldlace": (253, 245, 230),
    "antiquewhite": (250, 235, 215),
    "papayawhip": (255, 239, 213),
    "blanchedalmond": (255, 235, 205),
    "bisque": (255, 228, 196),
    "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173),
    "peachpuff": (255, 218, 185),
    "mistyrose": (255, 228, 225),
    "seashell": (255, 245, 238),
    "floralwhite": (255, 250, 240),
    "ghostwhite": (248, 248, 255),
    "honeydew": (240, 255, 240),
    "lightyellow": (255, 255, 224),
    "lemonchiffon": (255, 250, 205),
    "lightgoldenrodyellow": (250, 250, 210),
    "palegreen": (152, 251, 152),
    "mediumseagreen": (60, 179, 113),
    "darkturquoise": (0, 206, 209),
    "mediumturquoise": (72, 209, 204),
    "turquoise": (64, 224, 208),
    "cadetblue": (95, 158, 160),
    "powderblue": (176, 224, 230),
    "lightsteelblue": (176, 196, 222),
    "cornflowerblue": (100, 149, 237),
    "mediumslateblue": (123, 104, 238),
    "mediumvioletred": (199, 21, 133),
    "palevioletred": (219, 112, 147),
    "plum": (221, 160, 221),
    "orchid": (218, 112, 214),
    "thistle": (216, 191, 216),
    "darkviolet": (148, 0, 211),
    "blueviolet": (138, 43, 226),
    "dodgerblue": (30, 144, 255),
    "deepskyblue": (0, 191, 255),
    "lightskyblue": (135, 206, 250),
    "mediumaquamarine": (102, 205, 170),
    "aquamarine": (127, 255, 212),
    "darkcyan": (0, 139, 139),
    "darkseagreen": (143, 188, 143),
    "yellowgreen": (154, 205, 50),
    "olivedrab": (107, 142, 35),
    "darkolivegreen": (85, 107, 47),
    "mediumorchid": (186, 85, 211),
    "darkmagenta": (139, 0, 139),
    "hotpink": (255, 105, 180),
    "deeppink": (255, 20, 147),
    "lightpink": (255, 182, 193),
    "indianred": (205, 92, 92),
    "orangered": (255, 69, 0),
    "springgreen": (0, 255, 127),
    "mediumspringgreen": (0, 250, 154),
    "lawngreen": (124, 252, 0),
    "chartreuse": (127, 255, 0),
    "darkorchid": (153, 50, 204),
    "peru": (205, 133, 63),
    "burlywood": (222, 184, 135),
    "sandybrown": (244, 164, 96),
    "darkslategray": (47, 79, 79),
    "darkslategrey": (47, 79, 79),
    "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144),
    "lightslategray": (119, 136, 153),
    "lightslategrey": (119, 136, 153),
}


def parse_color(value: str | None) -> Color | None:
    """Parse a CSS color string; None for transparent/invalid (canvas port)."""
    if not value:
        return None
    s = str(value).strip()
    if not s or s == "transparent" or s == "none":
        return None
    if re.match(r"^rgba?\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*\)$", s):
        return None
    try:
        return _parse_impl(s)
    except (ValueError, OverflowError):
        return None


def _chan(v: str) -> float:
    if v.endswith("%"):
        return float(v[:-1]) / 100.0 * 255.0
    return float(v)


def _alpha(v: str | None) -> float:
    if v is None:
        return 1.0
    if v.endswith("%"):
        return float(v[:-1]) / 100.0
    return float(v)


def _parse_impl(s: str) -> Color:
    m = _HEX_RE.match(s)
    if m:
        h = m.group(1)
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h)
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        a = int(h[6:8], 16) / 255.0 if len(h) == 8 else 1.0
        return Color(r, g, b, a)
    m = _RGB_FN_RE.match(s)
    if m:
        return Color(_chan(m.group(1)), _chan(m.group(2)), _chan(m.group(3)), _alpha(m.group(4)))
    m = _HSL_FN_RE.match(s)
    if m:
        h = float(m.group(1))
        sat = float(m.group(2)) / 100.0
        light = float(m.group(3)) / 100.0
        c = (1 - abs(2 * light - 1)) * sat
        hp = h / 60.0
        x = c * (1 - abs(hp % 2 - 1))
        r = g = b = 0.0
        if hp < 1:
            r, g, b = c, x, 0.0
        elif hp < 2:
            r, g, b = x, c, 0.0
        elif hp < 3:
            r, g, b = 0.0, c, x
        elif hp < 4:
            r, g, b = 0.0, x, c
        elif hp < 5:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
        m_chan = light - c / 2
        return Color(
            round((r + m_chan) * 255),
            round((g + m_chan) * 255),
            round((b + m_chan) * 255),
            _alpha(m.group(4)),
        )
    named = _NAMED.get(s.lower())
    if named:
        return Color(*named)
    raise ValueError(f"unsupported color: {s}")


@dataclass(frozen=True, slots=True)
class HSL:
    h: float
    s: float
    light: float
    a: float = 1.0


def rgb_to_hsl(c: Color | None) -> HSL | None:
    """Port of the reference `rgbToHsl` (0-360 hue, 0-1 s/l)."""
    if not c:
        return None
    r = c.r / 255.0
    g = c.g / 255.0
    b = c.b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    h = 0.0
    s = 0.0
    light = (mx + mn) / 2.0
    if mx != mn:
        d = mx - mn
        s = d / (2 - mx - mn) if light > 0.5 else d / (mx + mn)
        if mx == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif mx == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6.0
    return HSL(h * 360.0, s, light, c.a)


def is_purple(c: Color | None) -> bool:
    """The 'VibeCode Purple' zone: indigo-violet hues with meaningful saturation."""
    if not c or c.a < 0.25:
        return False
    hsl = rgb_to_hsl(c)
    if not hsl:
        return False
    return 240 <= hsl.h <= 295 and hsl.s >= 0.35 and 0.15 < hsl.light < 0.85


def is_dark(c: Color | None) -> bool:
    if not c:
        return False
    hsl = rgb_to_hsl(c)
    return bool(hsl and hsl.light < 0.25)


def is_mid_grey(c: Color | None) -> bool:
    if not c:
        return False
    hsl = rgb_to_hsl(c)
    return bool(hsl and hsl.s < 0.15 and 0.35 < hsl.light < 0.75)


def relative_luminance(c: Color | None) -> float:
    """WCAG 2.x relative luminance over 0-255 channels."""
    if not c:
        return 0.0
    lin: list[float] = []
    for v in (c.r, c.g, c.b):
        s = v / 255.0
        lin.append(s / 12.92 if s <= 0.03928 else math.pow((s + 0.055) / 1.055, 2.4))
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(c1: Color | None, c2: Color | None) -> float:
    l1 = relative_luminance(c1)
    l2 = relative_luminance(c2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def channel_spread(c: Color | None) -> float:
    """Cheap chroma proxy: max(r,g,b) - min(r,g,b)."""
    if not c:
        return 0.0
    return max(c.r, c.g, c.b) - min(c.r, c.g, c.b)


def is_neutral(c: Color | None) -> bool:
    if not c or c.a < 0.05:
        return True
    return channel_spread(c) < 30.0


def effective_background(
    start: int,
    styles_of: Callable[[int], dict[str, str]],
    parent_of: Callable[[int], int | None],
) -> Color | None:
    """Walk ancestors for the first opaque background (reference port).

    ``start`` is the node index; ``styles_of(index) -> styles dict`` and
    ``parent_of(index) -> parent index or None``. Own bg, else nearest painted
    ancestor, else white as the document default (approx flag mirrors the
    reference's low-confidence gradient bail).
    """
    node: int | None = start
    guard = 0
    while node is not None and guard < 40:
        cs: dict[str, str] = styles_of(node)
        bg = parse_color(cs.get("background-color") or "")
        if bg and bg.a >= 0.5:
            return bg
        img: str = cs.get("background-image") or ""
        if img and img != "none" and re.search(r"gradient|url\(|image-set", img):
            stop = re.search(r"rgba?\([^)]+\)|#[0-9a-fA-F]{3,8}", img)
            if stop:
                sc = parse_color(stop.group(0))
                if sc and sc.a >= 0.5:
                    return sc
            return Color(255, 255, 255, 1.0, approx=True)
        node = parent_of(node)
    return Color(255, 255, 255, 1.0)
