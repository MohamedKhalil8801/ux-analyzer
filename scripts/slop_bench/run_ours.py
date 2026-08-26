"""Capture our detector's output per corpus URL.

Usage: python scripts/slop_bench/run_ours.py [slug ...]
Caches one JSON file per slug under scripts/slop_bench/ours/<slug>.json.
Invokes the real `uxa slop --json` CLI (same code path as report.html).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from corpus import CORPUS, OURS_DIR  # noqa: E402

VENV = Path(__file__).resolve().parents[2] / ".venv" / "Scripts"
UXA_EXE = VENV / "uxa.exe" if (VENV / "uxa.exe").is_file() else VENV / "uxa"


def capture(slug: str, url: str, force: bool = False) -> Path:
    out = ROOT / OURS_DIR / f"{slug}.json"
    if out.is_file() and not force:
        return out
    proc = subprocess.run(
        [str(UXA_EXE), "slop", url, "--json", "--copy"],
        capture_output=True,
        text=True,
        timeout=240,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"uxa slop failed for {url}: {proc.stderr[-500:] or proc.stdout[-500:]}"
        )
    data = json.loads(proc.stdout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"ours: {slug} score={data.get('score')} tier={data.get('tier')} flagged={data.get('patternsFlagged')}")
    return out


def main() -> None:
    slugs = sys.argv[1:] or [c["slug"] for c in CORPUS]
    by_slug = {c["slug"]: c for c in CORPUS}
    for slug in slugs:
        entry = by_slug[slug]
        capture(entry["slug"], entry["url"])


if __name__ == "__main__":
    main()
