#!/usr/bin/env python3
"""
Frozen-app entry point for GroundTruth (the packaged .exe).

Runs the FastAPI backend with the app OBJECT (not an import string, so there's no
reload machinery to confuse PyInstaller) and opens the UI in the default browser.
The console window doubles as a simple log; closing it stops the server.

For development, prefer `python run_gui.py` (identical behavior, import-string).
"""
from __future__ import annotations

import socket
import threading
import webbrowser

import uvicorn

from config import config
from server.app import APP_NAME, app


def _free_port(preferred: int) -> int:
    """Use the configured port if free, otherwise let the OS pick one."""
    for port in (preferred, 0):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((config.server_host, port))
            chosen = s.getsockname()[1]
            s.close()
            return chosen
        except OSError:
            continue
    return preferred


def main() -> int:
    port = _free_port(config.server_port)
    url = f"http://{config.server_host}:{port}"
    print("=" * 70)
    print(f"  {APP_NAME}  ·  real-time debate fact-checker")
    print(f"  Opening {url}")
    print(f"  Data + settings: %APPDATA%\\{APP_NAME}")
    print("  Close this window to quit.")
    print("=" * 70)

    threading.Timer(1.4, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=config.server_host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
