"""Back-to-back A/B capture of one URL to distinguish drift from logic gaps."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = Path("D:/Projects/ux-analyzer")
REF_RUNNER = BASE / "scripts/slop_bench/ref_runner"
UXA = BASE / ".venv/Scripts/uxa.exe"


def run_oracle(url: str) -> dict:
    proc = subprocess.run(
        ["node", "runner.mjs", url], capture_output=True, text=True, timeout=180, cwd=str(REF_RUNNER)
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"error": proc.stderr[-300:] or "empty"}
    return json.loads(proc.stdout)


def main() -> None:
    url = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for i in range(rounds):
        o = run_oracle(url)
        proc = subprocess.run(
            [str(UXA), "slop", url, "--json"], capture_output=True, text=True, timeout=240
        )
        m = json.loads(proc.stdout)
        if "error" in o:
            print(f"round {i}: oracle error {o['error'][:120]}")
            continue
        o_trig = sorted(p["id"] for p in o["patterns"] if p["triggered"])
        m_trig = sorted(p["id"] for p in m["patterns"] if p["triggered"])
        mark = "MATCH" if (o.get("score"), o_trig) == (m.get("score"), m_trig) else "DIFF"
        print(
            f"round {i} [{mark}]: oracle {o.get('score')}/{o.get('tier')} {o_trig}"
        )
        print(f"           ours  {m.get('score')}/{m.get('tier')} {m_trig}")


if __name__ == "__main__":
    main()
