"""
Persistent settings + spend tracking for GroundTruth.

Both live OUTSIDE the repo, in a per-user data dir (%APPDATA%\\GroundTruth on
Windows, ~/.groundtruth elsewhere), so the API key and your spend history are
never committed to git. The env var ANTHROPIC_API_KEY still works and takes
precedence at first launch if no key has been saved through the UI.

  settings.json   every editable knob (audio, STT, verdict, API, prices, budget)
  spend.json      cumulative money-spent log across every session
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

APP_NAME = "GroundTruth"

# Fields on Config the Settings/API tabs are allowed to write. API key handled separately.
_EDITABLE = [
    # audio / capture
    "audio_source", "audio_device_id", "sample_rate",
    "vad_aggressiveness", "vad_silence_ms", "vad_min_segment_ms", "vad_max_segment_s",
    # local STT
    "whisper_model_size", "whisper_device", "whisper_compute_type",
    # verdict calibration
    "min_sources",
    # concurrency / cache
    "max_concurrent_checks", "enable_cache",
    # server
    "server_host", "server_port",
    # budget
    "session_budget_usd",
    # API / judge
    "judge_model", "classifier_model", "web_search_tool_version", "max_search_uses",
    "credible_domains",
    # cost estimates
    "model_prices", "web_search_price_per_1k",
]


def data_dir() -> Path:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = Path(base) / APP_NAME if os.environ.get("APPDATA") else Path(base) / f".{APP_NAME.lower()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_path() -> Path:
    return data_dir() / "settings.json"


def _spend_path() -> Path:
    return data_dir() / "spend.json"


# ---------------------------------------------------------------------- #
#  Settings
# ---------------------------------------------------------------------- #
def load_settings() -> dict:
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    _settings_path().write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_settings_to_config(cfg, settings: dict) -> None:
    """Overlay saved settings onto the live Config. Only known fields are touched."""
    for key in _EDITABLE:
        if key in settings and settings[key] is not None:
            setattr(cfg, key, settings[key])
    # API key: saved key wins over env only if actually present.
    key = settings.get("anthropic_api_key")
    if key:
        cfg.anthropic_api_key = key


def update_and_persist(cfg, incoming: dict) -> dict:
    """Merge UI-submitted settings into the saved file + live config. Returns the
    UI-safe view. An empty/absent api key means 'keep the existing one'."""
    saved = load_settings()
    for key in _EDITABLE:
        if key in incoming:
            saved[key] = incoming[key]
    # De-dupe the allowlist so the web_search tool never sees duplicate domains.
    if isinstance(saved.get("credible_domains"), list):
        from ddld.factcheck.anthropic_judge import dedupe_domains
        saved["credible_domains"] = dedupe_domains(saved["credible_domains"])
    api_key = (incoming.get("anthropic_api_key") or "").strip()
    if api_key:
        saved["anthropic_api_key"] = api_key
    save_settings(saved)
    apply_settings_to_config(cfg, saved)
    return settings_for_ui(cfg, saved)


def settings_for_ui(cfg, saved: dict | None = None) -> dict:
    """Current values for the Settings/API tabs. The API key is never returned in
    full — only whether one is set and its last 4 chars."""
    if saved is None:
        saved = load_settings()
    out = {k: getattr(cfg, k) for k in _EDITABLE}
    key = cfg.anthropic_api_key or ""
    out["anthropic_api_key_set"] = bool(key)
    out["anthropic_api_key_hint"] = ("…" + key[-4:]) if len(key) >= 4 else ""
    out["anthropic_api_key_source"] = (
        "saved" if saved.get("anthropic_api_key") else ("env" if os.environ.get("ANTHROPIC_API_KEY") else "none")
    )
    out["data_dir"] = str(data_dir())
    return out


# ---------------------------------------------------------------------- #
#  Spend tracker (cumulative, across sessions)
# ---------------------------------------------------------------------- #
class SpendStore:
    """Append-only session records + running totals, persisted to spend.json."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        p = _spend_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"total_usd": 0.0, "sessions": []}

    def _flush(self) -> None:
        try:
            _spend_path().write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def record_session(self, *, cost_usd: float, source: str, claims: int,
                       web_searches: int, input_tokens: int, output_tokens: int,
                       started_at: float, ended_at: float) -> None:
        if cost_usd <= 0 and claims == 0:
            return  # nothing happened; don't clutter history
        with self._lock:
            self._data["sessions"].append({
                "ts": ended_at,
                "started_at": started_at,
                "duration_s": round(max(0.0, ended_at - started_at), 1),
                "source": source,
                "cost_usd": round(cost_usd, 4),
                "claims": claims,
                "web_searches": web_searches,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            self._data["total_usd"] = round(self._data.get("total_usd", 0.0) + cost_usd, 4)
            self._flush()

    def summary(self, limit: int = 50) -> dict:
        now = time.time()
        day_ago = now - 86400
        month_ago = now - 30 * 86400
        with self._lock:
            sessions = list(self._data.get("sessions", []))
            total = self._data.get("total_usd", 0.0)
        today = round(sum(s["cost_usd"] for s in sessions if s["ts"] >= day_ago), 4)
        month = round(sum(s["cost_usd"] for s in sessions if s["ts"] >= month_ago), 4)
        recent = sorted(sessions, key=lambda s: s["ts"], reverse=True)[:limit]
        return {
            "total_usd": round(total, 4),
            "today_usd": today,
            "month_usd": month,
            "session_count": len(sessions),
            "sessions": recent,
        }
