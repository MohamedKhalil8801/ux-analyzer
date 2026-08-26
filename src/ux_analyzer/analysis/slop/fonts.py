"""Slop-font fingerprint (port of @slop-detect/core `fonts.ts`).

The 2026 AI-generated-design font fingerprint: Inter dominates centered hero
headlines; Space Grotesk + Instrument Serif + Geist combos signal "tried to
look artsy".
"""

from __future__ import annotations

SLOP_FONT_PREFIXES = [
    "inter",  # Inter / Inter Display / Inter Tight
    "geist",  # Geist / Geist Sans / Geist Mono / GeistSans
    "space grotesk",
    "space-grotesk",
    "instrument serif",
    "instrument-serif",
    "general sans",
    "general-sans",
    "satoshi",
    "plus jakarta sans",
    "plus-jakarta",
    "plus jakarta",
    "manrope",  # 2026 AI-tool default
    "dm sans",  # shadcn/ui default
    "dm-sans",
    "cal sans",  # Linear/Vercel imitator hero font
    "cal-sans",
    "switzer",
    "clash display",
    "clash-display",
]

ACCENT_SERIF_PREFIXES = [
    "instrument serif",
    "instrument-serif",
    "fraunces",
    "playfair",
    "lora",
    "ibm plex serif",
    "spectral",
    "tinos",
    "cormorant",
]


def _first_family(family: str | None) -> str:
    if not family:
        return ""
    f = str(family).lower().replace('"', "").replace("'", "")
    return f.split(",")[0].strip()


def is_slop_font(family: str | None) -> bool:
    first = _first_family(family)
    if not first:
        return False
    return any(first.startswith(p) for p in SLOP_FONT_PREFIXES)


def is_accent_serif(family: str | None) -> bool:
    first = _first_family(family)
    if not first:
        return False
    return any(first.startswith(p) for p in ACCENT_SERIF_PREFIXES)
