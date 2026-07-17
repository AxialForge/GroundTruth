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
        try:
            config.validate()
        except RuntimeError as e:
            self._emit({"type": "status", "state": "error", "message": str(e)})
            return

        self.source = (params.get("source") or "loopback").lower()
        device_id = params.get("device_id") or ""
        file_path = params.get("file_path") or ""
        self._budget = float(params.get("budget_usd") or 0.0)

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
                    silence_ms=config.vad_silence_ms,
                    min_segment_ms=config.vad_min_segment_ms,
                    max_segment_s=config.vad_max_segment_s,
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
        self._emit({"type": "status", "state": "running", "message": f"Listening ({self.source})…"})
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
            elif action == "devices":
                await ws.send_json({"type": "devices", **list_devices()})
    except WebSocketDisconnect:
        session.stop()
    finally:
        send_task.cancel()


# Mount static assets (favicon, etc.) if any are added later.
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
