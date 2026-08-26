"""Deterministic bench: run oracle + ours on the SAME frozen HTML.

Usage: python scripts/slop_bench/bench_frozen.py [slug ...]
Writes oracle_frozen/<slug>.json and ours_frozen/<slug>.json. Run
compare.py --frozen after this.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from corpus import CORPUS  # noqa: E402

FROZEN_DIR = ROOT / "frozen"
ORACLE_OUT = ROOT / "oracle_frozen"
OURS_OUT = ROOT / "ours_frozen"
REF_RUNNER = ROOT / "ref_runner" / "runner.mjs"
UXA = ROOT.parents[1] / ".venv" / "Scripts" / "uxa.exe"


def file_url(path: Path) -> str:
    return "file:///" + str(path.resolve()).replace("\\", "/")


def stable_run(cmd: list[str], *, cwd: str | None = None, env=None, timeout: int) -> dict:
    """Run until two consecutive invocations agree on score/tier/triggered sets
    (CSS/network hiccups on frozen files make single-shot runs unreliable),
    max 4 attempts."""

    def sig(data: dict) -> tuple:
        triggered = tuple(p["id"] for p in data.get("patterns", []) if p["triggered"])
        copy = data.get("copy") or {}
        copy_trig = tuple(p["id"] for p in copy.get("patterns", []) if p["triggered"])
        return (data.get("score"), data.get("tier"), triggered, copy_trig)

    last: dict | None = None
    for _ in range(4):
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
        )
        if proc.returncode != 0:
            raise RuntimeError(f"command failed: {proc.stderr[-400:] or proc.stdout[-400:]}")
        data = json.loads(proc.stdout)
        if last is not None and sig(data) == sig(last):
            return data
        last = data
    if last is None:
        raise RuntimeError("no output")
    return last


def run(slug: str) -> None:
    html = FROZEN_DIR / f"{slug}.html"
    url = file_url(html)
    oracle = stable_run(
        ["node", str(REF_RUNNER), "--nojs", url],
        cwd=str(REF_RUNNER.parent),
        timeout=180,
    )
    ORACLE_OUT.mkdir(parents=True, exist_ok=True)
    (ORACLE_OUT / f"{slug}.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")

    env = dict(subprocess.os.environ)
    env["UXA_SLOP_DISABLE_JS"] = "1"
    ours = stable_run(
        [str(UXA), "slop", url, "--json", "--copy"],
        env=env,
        timeout=240,
    )
    OURS_OUT.mkdir(parents=True, exist_ok=True)
    (OURS_OUT / f"{slug}.json").write_text(json.dumps(ours, indent=2), encoding="utf-8")
    print(
        f"{slug}: oracle {oracle.get('score')}/{oracle.get('tier')} "
        f"ours {ours.get('score')}/{ours.get('tier')}"
    )


def main() -> None:
    slugs = sys.argv[1:] or [c["slug"] for c in CORPUS]
    for slug in slugs:
        if not (FROZEN_DIR / f"{slug}.html").is_file():
            print(f"skip {slug}: not frozen")
            continue
        run(slug)


if __name__ == "__main__":
    main()
