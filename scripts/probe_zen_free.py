"""Probe every zen free-tier model with a trivial prompt and report status.

Usage: python scripts/probe_zen_free.py
Reads UXA_LLM_API_KEY / UXA_LLM_SESSION_ID from .env.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://opencode.ai/zen/v1"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

MODELS = [
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "mimo-v2.5-free",
    "ling-3.0-flash-fin-free",
    "deepseek-v4-flash-free",
    "muse-spark-1.3-contributor-free",
    "muse-spark-1.2-contributor-free",
]

PROMPT = (
    "Choose one word from this list: alpha, beta, gamma. "
    'Return only JSON: {"kind": "<word>"}'
)


def env_value(name: str) -> str:
    text = Path(".env").read_text(encoding="utf-8")
    match = re.search(rf"^{name}=(.+)$", text, re.M)
    if not match:
        raise SystemExit(f"{name} not found in .env")
    return match.group(1).strip()


def main() -> None:
    key = env_value("UXA_LLM_API_KEY")
    session = env_value("UXA_LLM_SESSION_ID")
    headers = {
        "authorization": f"Bearer {key}",
        "x-opencode-session": session,
        "content-type": "application/json",
        "user-agent": UA,
        "accept": "application/json",
    }
    for model in MODELS:
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 2000,
            }
        ).encode()
        request = urllib.request.Request(
            f"{BASE}/chat/completions", data=body, headers=headers
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read())
            message = payload["choices"][0]["message"]
            content = message.get("content") or ""
            reasoned = bool(message.get("reasoning") or message.get("reasoning_content"))
            print(
                f"{model:32s} 200 {time.perf_counter() - started:6.1f}s "
                f"finish={payload['choices'][0].get('finish_reason')} "
                f"reasoned={reasoned} content={content[:80]!r}",
                flush=True,
            )
        except urllib.error.HTTPError as error:
            detail = error.read()[:200]
            print(
                f"{model:32s} {error.code} {time.perf_counter() - started:6.1f}s "
                f"{detail!r}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - probe reports everything
            print(
                f"{model:32s} ERR {type(error).__name__} {error}",
                flush=True,
            )
        time.sleep(1.0)


if __name__ == "__main__":
    main()
