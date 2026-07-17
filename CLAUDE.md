# CLAUDE.md — GroundTruth

Context for Claude Code working in this repo. Read before making changes.
(Product name is **GroundTruth**; repo `AxialForge/GroundTruth`. The package dir
is still `ddld/` for historical reasons — don't churn it.)

## What this is

A real-time fact-checker for spoken debates. It transcribes audio, extracts
factual claims, checks each against credible web sources, and labels it
GREEN / RED / YELLOW / GREY with citations. Exports the full log to `.docx` +
`.xlsx`. **Product name is "Lie Detector," but RED means "contradicted by the
cited evidence," never a bare "lie."**

The design goal is calibration and transparency, not confident accusation. A
false RED on a true claim, or a false GREEN on a false one, is the worst
outcome — bias toward YELLOW/GREY when evidence is thin.

## Current status

**Phase 0 (headless core) + Phase 1 (live desktop app) are built.** The pipeline
runs end to end — `audio/transcript → STT → claim extraction → fact-check judge →
verdict objects → .docx/.xlsx` — from a terminal (`run_headless.py`) or a live web
UI (`run_gui.py`) that captures system-loopback / mic audio, transcribes it with
streaming Whisper, and streams color-coded verdicts over a WebSocket. Do not
rewrite the core — extend it. Phase 1's remaining open item is PyInstaller `.exe`
packaging; next feature target is Phase 2 (Chrome extension) reusing the same
FastAPI backend + WebSocket contract.

Phase 1 was verified this far: server boots, all routes 200, real audio devices
enumerate, the web UI renders and the WebSocket connects. **Not yet exercised on
this machine:** a full live run (needs `ANTHROPIC_API_KEY` + `faster-whisper`
installed + audio actually playing + CUDA). Verify that path on first real use.

## Commands

```bash
pip install -r requirements.txt          # core + live: fastapi, uvicorn, soundcard, webrtcvad-wheels, faster-whisper
export ANTHROPIC_API_KEY=sk-ant-...       # required for extraction + judging
python run_headless.py --transcript samples/sample_debate.txt   # no audio needed
python run_headless.py --wav debate.wav --stt faster_whisper    # local STT from a .wav
python run_gui.py                         # live app -> http://127.0.0.1:8760 (loopback/mic/file)
```

There's no test suite yet. If you add one, use `pytest`, mock the Anthropic
client (see the smoke-test pattern: fake `messages.create` returning canned JSON;
a `Judge` subclass returning canned `CheckedClaim`s) so tests need no network.

## Architecture

```
config.py                 # ALL knobs: models, allowlist, thresholds, concurrency, audio, cost. Nothing hard-codes these elsewhere.
run_headless.py           # Phase 0 front-end: wires everything, terminal UI, exports on exit
run_gui.py                # Phase 1 front-end (dev): launches the FastAPI live web UI (import-string)
groundtruth_app.py        # Phase 1 front-end (frozen): entry point PyInstaller compiles into GroundTruth.exe
build/GroundTruth.spec    # PyInstaller one-folder build spec (webrtcvad excluded on purpose; see below)
installer/GroundTruth.iss # Inno Setup script -> GroundTruth-Setup.exe (Start Menu + desktop shortcuts)
ddld/
  types.py                # Verdict(enum) / Utterance / Claim / Source / CheckedClaim  <- the shared vocabulary
  pipeline.py             # Pipeline: orchestrator. Emits PENDING immediately, checks async, updates in place via callbacks
  cache.py                # ClaimCache: normalize + dedup so repeated claims aren't re-checked
  audio/                  # Phase 1 live capture (nothing else changed)
    base.py               # AudioCapture ABC: frames() -> Iterator[float32 mono PCM]; stop()
    soundcard_backend.py  # LoopbackCapture (WASAPI system audio) + MicrophoneCapture; list_devices()
    __init__.py           # make_capture(cfg, source, device_id) + list_devices()
  stt/
    base.py               # STTEngine ABC: stream() -> Iterator[Utterance]
    transcript_file.py    # replays a .txt transcript (test path, no audio)
    faster_whisper_stt.py # local STT from a .wav (private, no per-minute cost)
    streaming_whisper.py  # Phase 1: AudioCapture + webrtcvad segmentation -> live Utterance stream
    __init__.py           # make_stt(cfg, source) factory  (source may be an AudioCapture for streaming)
  claims/extractor.py     # ClaimExtractor: heuristic prefilter + cheap LLM classifier -> atomic claims
  factcheck/
    base.py               # Judge ABC: check(claim) -> CheckedClaim
    judge_prompt.py       # the strict, calibrated judge system prompt
    anthropic_judge.py    # AnthropicJudge: Messages API + web_search + code-side guardrails
  export/                 # export_docx / export_xlsx, color-coded, with disclaimer
server/                   # Phase 1 web front-end (same Pipeline callbacks the terminal uses)
  app.py                  # FastAPI: Session runs Pipeline on a thread; WebSocket /ws; /api/{devices,config,settings,spend}
  cost.py                 # CostMeter + MeteredAnthropic proxy + BudgetedJudge (session budget ceiling)
  store.py                # persistent settings + cumulative spend tracker in %APPDATA%\GroundTruth (secrets stay out of git)
  static/index.html       # tabbed UI — Live / Settings / API / Spending; GREY->color flip; click to expand sources
```

**Packaging notes.** The frozen build excludes `webrtcvad`: the webrtcvad-wheels
contrib hook is broken (`copy_metadata('webrtcvad')` fails on the -wheels dist), and
`StreamingWhisperSTT` already falls back to a pure-Python energy gate when webrtcvad
is absent (`_make_speech_gate`). One-folder build (native CTranslate2 DLLs); CPU STT
works out of the box, GPU needs CUDA libs on the machine. Exports + cache are rebased
to the data dir at startup so a read-only Program Files install still works.

**Phase 1 insertion points (nothing in the core was rewritten):** live audio is a
new `AudioCapture` feeding a new `StreamingWhisperSTT` (a third `STTEngine`,
registered in `make_stt`). The server subscribes to the same `on_utterance /
on_claim / on_verdict` callbacks as `run_headless.py`. Cost metering is a
non-invasive proxy around the Anthropic client; the budget ceiling is a `Judge`
decorator. If you extend Phase 1, keep these as the seams — don't reach into the core.

Data flows one way through plain dataclasses in `types.py`. Front-ends subscribe
to `Pipeline` callbacks (`on_utterance`, `on_claim`, `on_verdict`) — a desktop
GUI or extension backend consumes the same events the terminal runner does.

## Invariants — do NOT break these

1. **Calibration is enforced twice.** The prompt biases toward YELLOW/GREY, and
   `AnthropicJudge._calibrate()` re-checks in code: RED with no `stance=="contradicts"`
   source → downgrade to YELLOW; GREEN with fewer than `config.min_sources` distinct
   supporting domains → downgrade to YELLOW. Never remove this code-side check, and
   never let the model's raw verdict pass through unguarded.
2. **The allowlist is enforced server-side** via the web_search tool's
   `allowed_domains` (from `config.credible_domains`). Keep it there — don't
   fall back to unrestricted search.
3. **PENDING-first, async.** Claims must surface immediately as GREY and update in
   place when the check returns. Never block the transcript on a fact-check.
4. **The disclaimer ships in the UI and in every export.** See `config.DISCLAIMER`.
5. **Non-claims (opinion/prediction/question) stay GREY** and never trigger a
   search. The prefilter + classifier gate this; keep that gate.
6. **No secrets in code or the repo.** The API key comes from either the
   `ANTHROPIC_API_KEY` env var or the user's local settings file at
   `%APPDATA%\GroundTruth\settings.json` (written via the API tab). That file is
   per-user, outside the repo, and `.gitignore`d — never commit a key, never echo
   it back to the UI in full (see `store.settings_for_ui`, which masks to last-4).

## API specifics (verified current — don't regress to older values)

- Web search tool: `{"type": "web_search_20260318", "name": "web_search", "max_uses": N, "allowed_domains": [...]}`.
  Only ONE of `allowed_domains` / `blocked_domains` may be set.
- Judge model default `claude-sonnet-5`; fallback `claude-opus-4-8` if a model
  rejects web search. Classifier `claude-haiku-4-5-20251001`.
- Handle `stop_reason == "pause_turn"`: append the assistant turn and re-call until
  it settles (loop guard already in `anthropic_judge.py`).
- web_search responses contain multiple content block types; read the `text`
  blocks for the final JSON, don't index by position.

## Conventions

- Everything configurable lives in `config.py` (`Config` dataclass); read from
  `config` rather than passing loose params.
- New STT backend → subclass `STTEngine`, register in `stt/__init__.py:make_stt`.
- New fact-check backend (e.g. local model + search API) → subclass `Judge`; the
  pipeline doesn't care which one it holds.
- Fail open, not silent: if the classifier errors, treat candidates as checkable
  rather than dropping them (see `ClaimExtractor.extract`).
- Keep quotes in `Source.quote` short (≤ ~15 words) — attribution, not reproduction.

## Verdict model

| Verdict | Color | Rule |
|---------|-------|------|
| SUPPORTED | GREEN | ≥ `min_sources` (default 3) independent credible sources corroborate |
| CONTRADICTED | RED | directly refuted, REQUIRES a contradicting citation |
| DISPUTED | YELLOW | partial / cherry-picked / out of date / sources disagree |
| PENDING | GREY | check running, evidence thin, or not a checkable claim |

## Roadmap & next task

**Phase 1 — Live desktop app (BUILT). Decisions locked:**
- Front-end is a **local web UI** (FastAPI + `server/static/index.html`), chosen over
  native GUI because the same backend + WebSocket contract is reused by the Phase 2
  Chrome extension. Launch via `run_gui.py`.
- Audio is captured with **`soundcard`** — WASAPI system loopback (covers streaming,
  TV, any app) and microphone (in-person). `StreamingWhisperSTT` re-frames to 30 ms,
  segments with **webrtcvad**, and transcribes finalized segments with faster-whisper
  (set `whisper_device="cuda"` for the GPU). `Utterance`/`Pipeline` unchanged.
- Live UI: scrolling color feed, GREY→color flip in place, click-to-expand reasoning +
  source links, **live cost meter + optional per-session budget**, "Save log" button
  wired to the existing `export_docx`/`export_xlsx`.
- **Remaining:** PyInstaller `.exe`. Bundling CTranslate2/CUDA libs is finicky — do it
  once the app feels right. A `--onedir` build is far easier than `--onefile` here.

**Phase 2 — Chrome extension (MV3):** `chrome.tabCapture` for tab audio, reuse the
core via a local backend (FastAPI) or direct API calls, same UI language.

**Phase 3 — Polish:** source-list config UI, speaker diarization/labels, confidence
tuning, latency batching + prioritizing high-salience claims.

## Open decisions

Locked in Phase 1: front-end = local web UI (desktop-first, extension reuses the
backend later); model strategy = Claude-only; cost control = live meter + optional
per-session budget in the UI (prices in `config.model_prices`, marked estimates).

Still open:
- **PyInstaller `.exe`** packaging (the one Phase 1 to-do).
- **Speaker diarization** (who-said-what) — single-stream transcript for now.
- **Streaming Whisper tuning** — `vad_*` knobs in `config.py` control segment
  boundaries; if live segments come out too choppy or too laggy, tune there first.

## Note

Library/API details for live audio capture (WASAPI loopback packages, MV3
`tabCapture` semantics) move — verify current APIs at implementation time. The
Anthropic web_search + model details above were verified at build time; re-check
if you hit a version error.
