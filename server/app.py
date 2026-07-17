"""
Local web UI backend (Phase 1).

Runs the existing Pipeline and forwards its three callbacks — on_utterance,
on_claim, on_verdict — to the browser over a WebSocket. The core is untouched:
this file only wires audio/STT + judge into Pipeline and relays events, exactly
like run_headless.py does for the terminal.

The same WebSocket contract is what a future Chrome extension would speak to, so
this backend is reused, not thrown away, in Phase 2.

Launch with:  python run_gui.py     (from the repo root)
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import anthropic
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import config
from ddld import AnthropicJudge, ClaimCache, ClaimExtractor, Pipeline
from ddld.audio import list_devices, make_capture
from ddld.export import export_docx, export_xlsx
from ddld.stt.faster_whisper_stt import FasterWhisperSTT
from ddld.stt.streaming_whisper import StreamingWhisperSTT
from server.cost import BudgetedJudge, CostMeter, MeteredAnthropic
from server import store

APP_NAME = "GroundTruth"

# Listening modes tune only how audio is segmented — the fact-check pipeline is the
# same for a debate, a newscast, or one person talking. "speaker" = single speaker /
# speech / lecture: longer segments and more patient silence so a monologue isn't
# chopped mid-sentence. Debates get snappier cuts to keep up with crosstalk.
_MODE_PROFILES = {
    "debate":  {"silence_ms": 550, "max_segment_s": 12.0},
    "news":    {"silence_ms": 700, "max_segment_s": 14.0},
    "speaker": {"silence_ms": 900, "max_segment_s": 20.0},
    "podcast": {"silence_ms": 800, "max_segment_s": 18.0},
}

# Static assets: bundled next to the code, or under _MEIPASS when frozen by PyInstaller.
if getattr(sys, "frozen", False):
    _STATIC = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "server" / "static"
else:
    _STATIC = Path(__file__).parent / "static"

# Overlay any settings saved through the UI onto the live config at startup.
store.apply_settings_to_config(config, store.load_settings())

# Exports + the claim cache must live somewhere writable (a Program Files install
# is read-only), so keep them in the per-user data dir.
config.export_dir = str(store.data_dir() / "exports")
config.cache_path = str(store.data_dir() / "claim_cache.json")

# Cumulative money-spent tracker, shared across every session/connection.
SPEND = store.SpendStore()

app = FastAPI(title=APP_NAME)


# ---------------------------------------------------------------------- #
#  Message serialization
# ---------------------------------------------------------------------- #
def _claim_msg(c) -> dict:
    v = c.verdict
    return {
        "type": "claim",
        "id": c.claim.id,
        "utterance_id": c.claim.utterance_id,
        "speaker": c.claim.speaker or "",
        "text": c.claim.text,
        "checkable": c.claim.checkable,
        "verdict": v.value, "label": v.label, "hex": v.hex,
        "t": round(c.claim.t_start, 1),
    }


def _verdict_msg(c) -> dict:
    v = c.verdict
    return {
        "type": "verdict",
        "id": c.claim.id,
        "verdict": v.value, "label": v.label, "hex": v.hex,
        "confidence": round(c.confidence, 2),
        "reasoning": c.reasoning,
        "error": c.error or "",
        "sources": [
            {"url": s.url, "title": s.title, "quote": s.quote, "stance": s.stance}
            for s in c.sources
        ],
    }


# ---------------------------------------------------------------------- #
#  One capture/analyze session, tied to a single WebSocket connection.
# ---------------------------------------------------------------------- #
class Session:
    def __init__(self, ws: WebSocket, loop: asyncio.AbstractEventLoop, outbox: asyncio.Queue):
        self._ws = ws
        self._loop = loop
        self._outbox = outbox
        self._thread: threading.Thread | None = None
        self._capture = None
        self._pipe: Pipeline | None = None
        self._meter: CostMeter | None = None
        self._running = False
        self._started_at = 0.0
        self._budget = 0.0
        self.source = ""
        self._level_capture = None
        self._level_thread: threading.Thread | None = None
        self.mode = "debate"

    # --- called from worker threads: hand a message to the async sender ---
    def _emit(self, msg: dict) -> None:
        self._loop.call_soon_threadsafe(self._outbox.put_nowait, msg)

    def _stats(self) -> dict:
        cost = self._meter.snapshot() if self._meter else {}
        claims = self._pipe.checked_claims if self._pipe else []
        checkable = [c for c in claims if c.claim.checkable]
        pending = [c for c in checkable if c.checked_at is None and not c.error]
        return {
            "type": "stats",
            **cost,
            "budget_usd": self._budget,
            "over_budget": bool(self._budget and cost.get("cost_usd", 0) >= self._budget),
            "claims_total": len(checkable),
            "claims_pending": len(pending),
        }

    # ------------------------------------------------------------------ #
    def start(self, params: dict) -> None:
        if self._running:
            self._emit({"type": "status", "state": "running", "message": "Already running."})
            return
        self.level_stop()  # release the audio device from any level-test meter
        try:
            config.validate()
        except RuntimeError as e:
            self._emit({"type": "status", "state": "error", "message": str(e)})
            return

        self.source = (params.get("source") or "loopback").lower()
        device_id = params.get("device_id") or ""
        file_path = params.get("file_path") or ""
        self._budget = float(params.get("budget_usd") or 0.0)
        self.mode = (params.get("mode") or "debate").lower()
        speaker = (params.get("speaker") or "").strip() or None
        prof = _MODE_PROFILES.get(self.mode, {})
        silence_ms = int(prof.get("silence_ms", config.vad_silence_ms))
        max_segment_s = float(prof.get("max_segment_s", config.vad_max_segment_s))

        # Metered client so every extractor + judge call lands on the cost meter.
        self._meter = CostMeter(config.model_prices, config.web_search_price_per_1k)
        raw_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        client = MeteredAnthropic(raw_client, self._meter)

        # STT source: live capture (loopback/mic) or a finished .wav file.
        try:
            if self.source == "file":
                if not file_path or not os.path.exists(file_path):
                    raise RuntimeError(f"File not found: {file_path!r}")
                stt = FasterWhisperSTT(
                    file_path, model_size=config.whisper_model_size,
                    device=config.whisper_device, compute_type=config.whisper_compute_type,
                    speaker=speaker,
                )
                self._capture = None
            else:
                self._capture = make_capture(config, self.source, device_id)
                stt = StreamingWhisperSTT(
                    self._capture,
                    model_size=config.whisper_model_size,
                    device=config.whisper_device,
                    compute_type=config.whisper_compute_type,
                    sample_rate=config.sample_rate,
                    vad_aggressiveness=config.vad_aggressiveness,
                    silence_ms=silence_ms,
                    min_segment_ms=config.vad_min_segment_ms,
                    max_segment_s=max_segment_s,
                    speaker=speaker,
                )
        except Exception as e:
            self._emit({"type": "status", "state": "error", "message": f"Audio setup failed: {e}"})
            return

        extractor = ClaimExtractor(client, config.classifier_model)
        judge = BudgetedJudge(AnthropicJudge(client, config), self._meter, self._budget)
        cache = ClaimCache(config.cache_path, enabled=config.enable_cache)

        pipe = Pipeline(
            stt=stt, extractor=extractor, judge=judge, cache=cache,
            max_workers=config.max_concurrent_checks,
        )
        pipe.on_utterance = lambda u: self._emit({
            "type": "utterance", "id": u.id, "t": round(u.start, 1),
            "speaker": u.speaker or "", "text": u.text,
        })

        def on_claim(c):
            self._emit(_claim_msg(c))
            self._emit(self._stats())

        def on_verdict(c):
            self._emit(_verdict_msg(c))
            self._emit(self._stats())

        pipe.on_claim = on_claim
        pipe.on_verdict = on_verdict
        self._pipe = pipe

        self._running = True
        self._started_at = time.time()
        who = f" · {speaker}" if speaker else ""
        self._emit({"type": "status", "state": "running",
                    "message": f"Listening — {self.mode}{who} ({self.source})…"})
        self._thread = threading.Thread(target=self._run, args=(cache,), daemon=True)
        self._thread.start()

    def _run(self, cache: ClaimCache) -> None:
        try:
            self._pipe.run()
        except Exception as e:  # pragma: no cover
            self._emit({"type": "status", "state": "error", "message": f"Pipeline error: {e}"})
        finally:
            try:
                cache.flush()
            except Exception:
                pass
            self._persist_spend()
            self._running = False
            self._emit(self._stats())
            self._emit({"type": "spend", **SPEND.summary()})
            self._emit({"type": "status", "state": "stopped", "message": "Stopped."})

    def _persist_spend(self) -> None:
        if not self._meter or not self._pipe:
            return
        snap = self._meter.snapshot()
        checkable = [c for c in self._pipe.checked_claims if c.claim.checkable]
        SPEND.record_session(
            cost_usd=snap.get("cost_usd", 0.0),
            source=self.source,
            claims=len(checkable),
            web_searches=snap.get("web_searches", 0),
            input_tokens=snap.get("input_tokens", 0),
            output_tokens=snap.get("output_tokens", 0),
            started_at=self._started_at or time.time(),
            ended_at=time.time(),
        )

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.stop()   # ends the STT stream; pipeline drains in-flight checks then returns
        # File mode has no live capture to interrupt; it finishes on its own.

    # ------------------------------------------------------------------ #
    #  Audio level meter (no fact-checking) — a quick "is audio coming in?" test.
    # ------------------------------------------------------------------ #
    def level_start(self, params: dict) -> None:
        if self._running:
            self._emit({"type": "level_status", "state": "error", "message": "Stop the session first."})
            return
        self.level_stop()
        source = (params.get("source") or "loopback").lower()
        if source == "file":
            self._emit({"type": "level_status", "state": "error", "message": "Pick a live source (system audio or mic) to test."})
            return
        try:
            cap = make_capture(config, source, params.get("device_id") or "")
        except Exception as e:
            self._emit({"type": "level_status", "state": "error", "message": f"Audio: {e}"})
            return
        self._level_capture = cap
        self._emit({"type": "level_status", "state": "on", "message": f"Metering {source}…"})
        self._level_thread = threading.Thread(target=self._level_run, args=(cap,), daemon=True)
        self._level_thread.start()

    def _level_run(self, cap) -> None:
        import math
        import numpy as np
        try:
            for block in cap.frames():
                if block is None or len(block) == 0:
                    continue
                rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float32)) + 1e-12))
                peak = float(np.max(np.abs(block)))
                db = 20.0 * math.log10(rms + 1e-9)
                self._emit({"type": "level", "rms": round(rms, 5), "peak": round(peak, 5), "db": round(db, 1)})
        except Exception as e:  # pragma: no cover
            self._emit({"type": "level_status", "state": "error", "message": f"Audio error: {e}"})
        finally:
            self._emit({"type": "level_status", "state": "off", "message": "Meter stopped."})

    def level_stop(self) -> None:
        if self._level_capture is not None:
            self._level_capture.stop()
            self._level_capture = None

    def save(self) -> dict:
        if self._pipe is None:
            return {"type": "saved", "error": "Nothing to save yet."}
        os.makedirs(config.export_dir, exist_ok=True)
        base = time.strftime("session_%Y%m%d_%H%M%S")
        docx_path = os.path.join(config.export_dir, base + ".docx")
        xlsx_path = os.path.join(config.export_dir, base + ".xlsx")
        try:
            export_docx(docx_path, self._pipe.transcript, self._pipe.checked_claims, config.DISCLAIMER)
            export_xlsx(xlsx_path, self._pipe.transcript, self._pipe.checked_claims, config.DISCLAIMER)
        except Exception as e:
            return {"type": "saved", "error": str(e)}
        return {"type": "saved", "docx": os.path.abspath(docx_path), "xlsx": os.path.abspath(xlsx_path)}


# ---------------------------------------------------------------------- #
#  Routes
# ---------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/devices")
def devices():
    return JSONResponse(list_devices())


@app.get("/api/config")
def ui_config():
    return JSONResponse({
        "app_name": APP_NAME,
        "disclaimer": config.DISCLAIMER,
        "min_sources": config.min_sources,
        "default_source": config.audio_source,
        "budget_usd": config.session_budget_usd,
    })


@app.get("/api/settings")
def get_settings():
    return JSONResponse(store.settings_for_ui(config))


@app.post("/api/settings")
async def post_settings(request: Request):
    incoming = await request.json()
    view = store.update_and_persist(config, incoming)
    # Some fields (server_host/port) only take effect on restart; flag that.
    view["restart_required_for"] = ["server_host", "server_port"]
    return JSONResponse({"ok": True, "settings": view})


@app.get("/api/spend")
def get_spend():
    return JSONResponse(SPEND.summary())


def _short_err(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "401" in msg or "authentication" in low or "invalid x-api-key" in low or "invalid api key" in low:
        return "Authentication failed — the API key is missing or wrong."
    if "web_search" in low or "does not support" in low:
        return f"Model rejected web search: {msg[:180]}"
    if "rate" in low and "limit" in low:
        return "Rate limited — try again in a moment."
    return msg[:220]


@app.post("/api/test/anthropic")
def test_anthropic():
    """Live check: does the saved key work, and does the judge model accept
    web_search with the allowlist? Costs a few cents (one search + a tiny ping)."""
    key = config.anthropic_api_key
    if not key:
        return JSONResponse({"ok": False, "message": "No API key set. Add one above (or set ANTHROPIC_API_KEY)."})
    client = anthropic.Anthropic(api_key=key)
    out = {"key_source": store.settings_for_ui(config).get("anthropic_api_key_source", "?")}

    # 1) key + connectivity, via the cheap classifier model
    t = time.time()
    try:
        r = client.messages.create(
            model=config.classifier_model, max_tokens=5,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        )
        reply = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
        out["classifier"] = {"ok": True, "latency_ms": int((time.time() - t) * 1000),
                             "model": config.classifier_model, "reply": reply[:20]}
    except Exception as e:
        out["classifier"] = {"ok": False, "model": config.classifier_model, "error": _short_err(e)}

    # 2) judge model + web_search + allowlist (the feature that actually matters)
    t = time.time()
    try:
        tool = {"type": config.web_search_tool_version, "name": "web_search", "max_uses": 1}
        if config.credible_domains:
            tool["allowed_domains"] = config.credible_domains
        msgs = [{"role": "user", "content":
                 "Use web search to find the current U.S. unemployment rate, then reply in one short sentence with the number and its source."}]
        r = client.messages.create(model=config.judge_model, max_tokens=500, tools=[tool], messages=msgs)
        guard = 0
        while getattr(r, "stop_reason", "") == "pause_turn" and guard < 3:
            msgs.append({"role": "assistant", "content": r.content})
            r = client.messages.create(model=config.judge_model, max_tokens=500, tools=[tool], messages=msgs)
            guard += 1
        searched = 0
        st = getattr(getattr(r, "usage", None), "server_tool_use", None)
        if st is not None:
            searched = int(getattr(st, "web_search_requests", 0) or 0)
        reply = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
        out["judge"] = {"ok": True, "latency_ms": int((time.time() - t) * 1000),
                        "model": config.judge_model, "searched": searched, "reply": reply[:200]}
    except Exception as e:
        out["judge"] = {"ok": False, "model": config.judge_model, "error": _short_err(e)}

    out["ok"] = bool(out.get("classifier", {}).get("ok"))
    return JSONResponse(out)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()
    session = Session(ws, loop, outbox)

    async def sender():
        while True:
            msg = await outbox.get()
            await ws.send_json(msg)

    send_task = asyncio.create_task(sender())
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            if action == "start":
                session.start(data)
            elif action == "stop":
                session.stop()
            elif action == "save":
                await ws.send_json(session.save())
            elif action == "level_start":
                session.level_start(data)
            elif action == "level_stop":
                session.level_stop()
            elif action == "devices":
                await ws.send_json({"type": "devices", **list_devices()})
    except WebSocketDisconnect:
        session.stop()
        session.level_stop()
    finally:
        send_task.cancel()


# Mount static assets (favicon, etc.) if any are added later.
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
