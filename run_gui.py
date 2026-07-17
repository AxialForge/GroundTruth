#!/usr/bin/env python3
"""
Phase 1 launcher: the live desktop app (local web UI).

Starts the FastAPI backend and opens the feed in your browser. Pick an audio
source (system loopback / microphone / a .wav file), hit Start, and watch claims
appear GREY and flip color as each fact-check returns.

    pip install -r requirements.txt          # now includes fastapi/uvicorn/soundcard/etc.
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_gui.py

Everything downstream of audio is the same core the headless runner uses.
"""
from __future__ import annotations

import sys
import threading
import webbrowser

import uvicorn

from config import config


def main() -> int:
    try:
        config.validate()
    except RuntimeError as e:
        # Not fatal for launching — the UI will surface it on Start — but warn early.
        print(f"[warn] {e}", file=sys.stderr)

    url = f"http://{config.server_host}:{config.server_port}"
    print("=" * 70)
    print("  GROUNDTRUTH  ·  live debate fact-checker")
    print(f"  Opening {url}")
    print(f"  {config.DISCLAIMER}")
    print("=" * 70)

    # Open the browser once the server is (almost certainly) up.
    threading.Timer(1.3, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "server.app:app",
        host=config.server_host,
        port=config.server_port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
