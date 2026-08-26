"""Benchmark corpus for slop-detector parity: clean vs sloppy landing pages.

Each entry: slug, URL, expected_lean (what a correct detector should lean
toward — used by the critic only as a sanity anchor, never as the verdict).
"""

CORPUS = [
    # ── Clean / human-crafted ────────────────────────────────────────────────
    {"slug": "hackernews", "url": "https://news.ycombinator.com", "lean": "clean"},
    {"slug": "wikipedia", "url": "https://en.wikipedia.org", "lean": "clean"},
    {"slug": "stripe", "url": "https://stripe.com", "lean": "clean"},
    {"slug": "apple", "url": "https://www.apple.com", "lean": "clean"},
    {"slug": "smashingmagazine", "url": "https://www.smashingmagazine.com", "lean": "clean"},
    {"slug": "github", "url": "https://github.com", "lean": "clean"},
    {"slug": "nytimes", "url": "https://www.nytimes.com", "lean": "clean"},
    # ── Edge cases: premium sites that historically trigger FPs ─────────────
    {"slug": "linear", "url": "https://linear.app", "lean": "clean"},
    {"slug": "vercel", "url": "https://vercel.com", "lean": "clean"},
    # ── AI-builder / template slop ───────────────────────────────────────────
    {"slug": "bolt", "url": "https://bolt.new", "lean": "sloppy"},
    {"slug": "v0", "url": "https://v0.dev", "lean": "sloppy"},
    {"slug": "builder", "url": "https://www.builder.io", "lean": "sloppy"},
    {"slug": "relume", "url": "https://relume.io", "lean": "sloppy"},
    {"slug": "lovable", "url": "https://lovable.dev", "lean": "sloppy"},
    {"slug": "shipfast", "url": "https://shipfa.st", "lean": "sloppy"},
    {"slug": "cursor", "url": "https://www.cursor.com", "lean": "sloppy"},
]

ORACLE_DIR = "oracle"
OURS_DIR = "ours"

CLEAN_SLUGS = [c["slug"] for c in CORPUS if c["lean"] == "clean"]
SLOPPY_SLUGS = [c["slug"] for c in CORPUS if c["lean"] == "sloppy"]
