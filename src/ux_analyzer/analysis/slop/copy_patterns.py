"""The 9 copy-slop patterns (port of @slop-detect/core `copyPatterns.ts`).

Pure text analysis over the extracted text context
``{text, headings, paragraphs, wordCount}``. Fires on density, not single
occurrences — the reference's calibration philosophy.
"""

from __future__ import annotations

import re
from typing import Any

BUZZWORDS = [
    "delve",
    "leverage",
    "leveraging",
    "seamless",
    "seamlessly",
    "robust",
    "elevate",
    "unlock",
    "unlocking",
    "empower",
    "empowering",
    "streamline",
    "streamlined",
    "harness",
    "harnessing",
    "foster",
    "fostering",
    "cutting-edge",
    "game-changer",
    "game-changing",
    "revolutionize",
    "revolutionary",
    "transformative",
    "innovative",
    "holistic",
    "synergy",
    "paradigm",
    "tapestry",
    "realm",
    "landscape",
    "navigate",
    "navigating",
    "embark",
    "bespoke",
    "curated",
    "meticulous",
    "meticulously",
    "unparalleled",
    "unprecedented",
    "supercharge",
    "turbocharge",
    "next-level",
    "best-in-class",
    "world-class",
    "state-of-the-art",
    "frictionless",
    "effortless",
    "effortlessly",
    "comprehensive",
    "ever-evolving",
    "fast-paced",
    "dynamic",
    "pivotal",
    "underscore",
    "underscores",
    "testament",
    "beacon",
    "plethora",
    "myriad",
]

COPY_PATTERNS: list[dict[str, Any]] = []


def _register(p: dict[str, Any]) -> None:
    COPY_PATTERNS.append(p)


def _count(text: str, re_obj: re.Pattern[str]) -> int:
    m = re_obj.findall(text)
    return len(m)


def _unique(text: str, re_obj: re.Pattern[str], max_n: int = 5) -> list[str]:
    out: set[str] = set()
    for m in re_obj.finditer(text):
        out.add((m.group(0) or "").lower().strip())
        if len(out) >= 50:
            break
    return list(out)[:max_n]


def _per1000(count: int, word_count: int) -> float:
    if not word_count:
        return 0.0
    return (count / word_count) * 1000.0


# ── C1. BUZZWORD DENSITY ────────────────────────────────────────────────────
def _buzzword_density(ctx: dict[str, Any]) -> dict[str, Any]:
    text = (ctx.get("text") or "").lower()
    word_count = ctx.get("wordCount") or 0
    hits: list[dict[str, Any]] = []
    total = 0
    for w in BUZZWORDS:
        esc = re.escape(w)
        re_obj = re.compile(rf"(?:^|[^a-zA-Z-]){esc}(?:[^a-zA-Z-]|$)", re.IGNORECASE)
        c = _count(text, re_obj)
        if c > 0:
            hits.append({"word": w, "count": c})
            total += c
    hits.sort(key=lambda h: h["count"], reverse=True)
    density = _per1000(total, word_count)
    distinct = len(hits)
    return {
        "total": total,
        "distinct": distinct,
        "density": round(density * 10) / 10,
        "topWords": hits[:6],
        "triggered": distinct >= 4 or (total >= 3 and density >= 6),
    }


# ── C2. EM-DASH OVERLOAD ────────────────────────────────────────────────────
def _em_dash_overload(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text") or ""
    word_count = ctx.get("wordCount") or 0
    em = _count(text, re.compile("\u2014"))
    en = _count(text, re.compile(r"\s\u2013\s"))
    total = em + en
    density = _per1000(total, word_count)
    return {
        "emDashes": em,
        "enDashAsDash": en,
        "total": total,
        "density": round(density * 10) / 10,
        "triggered": (total >= 4 and density >= 7) or em >= 8,
    }


# ── C3. ANTITHESIS "not just X, it's Y" ─────────────────────────────────────
def _antithesis(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text") or ""
    patterns = [
        re.compile(r"\bit['’]?s not just\b", re.IGNORECASE),
        re.compile(r"\bnot just (?:a|an|about)\b", re.IGNORECASE),
        re.compile(r"\bisn['’]?t just\b", re.IGNORECASE),
        re.compile(r"\bnot only\b[^.?!]{1,80}\bbut also\b", re.IGNORECASE),
        re.compile(r"\bmore than just\b", re.IGNORECASE),
        re.compile(r"\bthis is(?:n['’]?t)? (?:just )?about\b", re.IGNORECASE),
    ]
    total = 0
    samples: list[str] = []
    for re_obj in patterns:
        total += _count(text, re_obj)
        for f in _unique(text, re_obj, 2):
            if len(samples) < 4:
                samples.append(f)
    return {"total": total, "samples": samples, "triggered": total >= 2}


# ── C4. FILLER OPENERS ──────────────────────────────────────────────────────
def _filler_openers(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text") or ""
    phrases = [
        re.compile(r"\bin today['’]?s (?:fast-paced|ever-evolving|digital|modern|competitive)\b", re.IGNORECASE),
        re.compile(r"\bin the world of\b", re.IGNORECASE),
        re.compile(r"\bin an era (?:of|where)\b", re.IGNORECASE),
        re.compile(r"\bwhen it comes to\b", re.IGNORECASE),
        re.compile(r"\bat the end of the day\b", re.IGNORECASE),
        re.compile(r"\bin the realm of\b", re.IGNORECASE),
        re.compile(r"\bin the ever-(?:evolving|changing|expanding)\b", re.IGNORECASE),
    ]
    total = 0
    samples: list[str] = []
    for re_obj in phrases:
        total += _count(text, re_obj)
        for f in _unique(text, re_obj, 1):
            if len(samples) < 4:
                samples.append(f)
    return {"total": total, "samples": samples, "triggered": total >= 1}


# ── C5. FORMULAIC CLOSERS ───────────────────────────────────────────────────
def _formulaic_closers(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text") or ""
    re_obj = re.compile(
        r"\b(?:in conclusion|in summary|to sum up|to summarize|all in all|ultimately,|in the end,|when all is said and done)\b",
        re.IGNORECASE,
    )
    total = _count(text, re_obj)
    return {"total": total, "samples": _unique(text, re_obj, 3), "triggered": total >= 1}


# ── C6. RULE-OF-THREE TRICOLON ──────────────────────────────────────────────
def _rule_of_three(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text") or ""
    re_obj = re.compile(r"\b([a-z]{4,15}),\s+([a-z]{4,15}),\s+and\s+([a-z]{4,15})\b", re.IGNORECASE)
    total = _count(text, re_obj)
    samples = _unique(text, re_obj, 3)
    re2 = re.compile(r"\b([a-z]{4,15}),\s+([a-z]{4,15}),\s+([a-z]{4,15})\.\s", re.IGNORECASE)
    total2 = _count(text, re2)
    return {"total": total + total2, "samples": samples, "triggered": total + total2 >= 3}


# ── C7. "WHETHER YOU'RE … OR …" ─────────────────────────────────────────────
def _whether_youre(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text") or ""
    re_obj = re.compile(r"\bwhether you['’]?re\b[^.?!]{1,80}\bor\b", re.IGNORECASE)
    total = _count(text, re_obj)
    return {"total": total, "samples": _unique(text, re_obj, 2), "triggered": total >= 1}


# ── C8. INVISIBLE / SMART UNICODE ARTIFACTS ─────────────────────────────────
def _unicode_artifacts(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text") or ""
    zero_width = _count(text, re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]"))
    nbsp = _count(text, re.compile("\u00A0"))
    narrow_nbsp = _count(text, re.compile("\u202F"))
    math_alnum = 0
    for ch in text:
        cp = ord(ch)
        if 0x1D400 <= cp <= 0x1D7FF:
            math_alnum += 1
    total = zero_width + narrow_nbsp + math_alnum + (1 if nbsp > 6 else 0)
    return {
        "zeroWidth": zero_width,
        "narrowNbsp": narrow_nbsp,
        "nbsp": nbsp,
        "mathAlnum": math_alnum,
        "total": total,
        "triggered": zero_width >= 1 or narrow_nbsp >= 1 or math_alnum >= 4,
    }


# ── C9. EMOJI BULLET HEADERS ────────────────────────────────────────────────
def _emoji_bullets(ctx: dict[str, Any]) -> dict[str, Any]:
    # Reference lead emoji set (surrogate pairs resolved to code points so the
    # Python class actually matches astral-plane emoji).
    lead = re.compile(
        r"^\s*(?:[\u2705\u2728\u2734\u2733\u2B50\U0001F680\U0001F525\U0001F4A1\U0001F389\U0001F44D\U0001F64C\u26A1\U0001F4AA\U0001F6E0\U0001F449])",
        re.UNICODE,
    )
    lines = list(ctx.get("headings") or []) + list(ctx.get("paragraphs") or [])
    count = 0
    samples: list[str] = []
    for line in lines:
        if lead.match(line):
            count += 1
            if len(samples) < 3:
                samples.append(line[:40])
    return {"count": count, "samples": samples, "triggered": count >= 3}


# ── Pattern registry ────────────────────────────────────────────────────────
_register(
    {
        "id": "buzzword_density",
        "label": "AI buzzword density (leverage / seamless / robust / elevate …)",
        "short": "Buzzwords",
        "category": "copy",
        "weight": 7,
        "match": _buzzword_density,
    }
)
_register(
    {
        "id": "em_dash_overload",
        "label": "Em-dash overload (— used as the default connective)",
        "short": "Em-dashes",
        "category": "copy",
        "weight": 5,
        "match": _em_dash_overload,
    }
)
_register(
    {
        "id": "antithesis_construction",
        "label": '"It\'s not just X — it\'s Y" / "not only … but also" antithesis',
        "short": "Not-just-X",
        "category": "copy",
        "weight": 6,
        "match": _antithesis,
    }
)
_register(
    {
        "id": "filler_openers",
        "label": 'Generic filler openers ("In today\'s fast-paced world …")',
        "short": "Filler opener",
        "category": "copy",
        "weight": 5,
        "match": _filler_openers,
    }
)
_register(
    {
        "id": "formulaic_closers",
        "label": 'Essay-style closers ("In conclusion", "In summary", "Ultimately")',
        "short": "Closers",
        "category": "copy",
        "weight": 4,
        "match": _formulaic_closers,
    }
)
_register(
    {
        "id": "rule_of_three",
        "label": 'Rule-of-three adjective tricolons ("fast, reliable, and scalable")',
        "short": "Rule of three",
        "category": "copy",
        "weight": 4,
        "match": _rule_of_three,
    }
)
_register(
    {
        "id": "whether_youre",
        "label": '"Whether you\'re X or Y" audience-spanning construction',
        "short": "Whether-you",
        "category": "copy",
        "weight": 3,
        "match": _whether_youre,
    }
)
_register(
    {
        "id": "unicode_artifacts",
        "label": "Invisible Unicode + smart-quote artifacts (copy-pasted from an LLM)",
        "short": "Unicode tells",
        "category": "copy",
        "weight": 4,
        "match": _unicode_artifacts,
    }
)
_register(
    {
        "id": "emoji_bullet_headers",
        "label": "Emoji-prefixed bullet/headers (🚀 ✅ ✨ as list markers)",
        "short": "Emoji bullets",
        "category": "copy",
        "weight": 3,
        "match": _emoji_bullets,
    }
)


def run_copy_patterns(text_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Run the 9 copy patterns; returns rows [{id,label,short,weight,triggered,evidence}]."""
    ctx = text_context or {"text": "", "headings": [], "paragraphs": [], "wordCount": 0}
    rows: list[dict[str, Any]] = []
    for p in COPY_PATTERNS:
        try:
            evidence = p["match"](ctx)
        except Exception as exc:  # noqa: BLE001
            evidence = {"triggered": False, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(
            {
                "id": p["id"],
                "label": p["label"],
                "short": p["short"],
                "axis": "copy",
                "category": p["category"],
                "weight": p["weight"],
                "triggered": bool(evidence.get("triggered")),
                "evidence": evidence,
            }
        )
    return rows
