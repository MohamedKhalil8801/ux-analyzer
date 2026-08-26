"""Capture ground-truth oracle JSON from the PUBLISHED reference engine.

Runs scripts/slop_bench/ref_runner/runner.mjs — a thin driver around the
published @slop-detect/core 0.5.1 (the exact code behind `npx slop-detect`)
that also emits the copy axis, which the published CLI drops from --json.

Usage: python scripts/slop_bench/capture_oracle.py [slug ...]
Caches one JSON file per slug under scripts/slop_bench/oracle/<slug>.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from corpus import CORPUS, ORACLE_DIR  # noqa: E402

REF_RUNNER = ROOT / "ref_runner" / "runner.mjs"


def capture(slug: str, url: str, force: bool = False) -> Path:
    out = ROOT / ORACLE_DIR / f"{slug}.json"
    if out.is_file() and not force:
        return out
    proc = subprocess.run(
        ["node", str(REF_RUNNER), url],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REF_RUNNER.parent),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ref runner failed for {url}: {proc.stderr[-500:]}")
    data = json.loads(proc.stdout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"oracle: {slug} score={data.get('score')} tier={data.get('tier')} flagged={data.get('patternsFlagged')}")
    return out


def main() -> None:
    slugs = sys.argv[1:] or [c["slug"] for c in CORPUS]
    by_slug = {c["slug"]: c for c in CORPUS}
    for slug in slugs:
        entry = by_slug[slug]
        capture(entry["slug"], entry["url"])


if __name__ == "__main__":
    main()
