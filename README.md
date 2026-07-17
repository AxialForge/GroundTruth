# GroundTruth

**Real-time debate fact-checker.** GroundTruth listens to spoken audio, transcribes
it, pulls out factual claims, checks each against credible web sources, and returns a
calibrated, cited verdict — GREEN / RED / YELLOW / GREY. Exports the full log to
`.docx` and `.xlsx`.

> **RED means "contradicted by the cited evidence," not "lie."** The whole design
> goal is calibration and transparency, not confident accusation.

**Phase 0** is the headless core, proven in a terminal (`run_headless.py`).
**Phase 1** is the live desktop app (`run_gui.py` / the packaged `GroundTruth.exe`) —
a local web UI with four tabs:

- **Live** — pick a source (system audio, mic, or a `.wav`) and a **listening mode**
  (debate / news / single speaker / podcast), optionally set a **speaker label**, then
  press Start. Claims appear grey and flip color as each fact-check returns; click one
  for reasoning + sources. Works the same for a debate, a newscast, or one person talking —
  the mode just tunes how audio is segmented, and a hallucination guard keeps music/applause
  in continuous audio from becoming false claims.
- **Settings** — audio/VAD, Whisper model + GPU, verdict threshold, concurrency, budget.
- **API** — Anthropic key (stored locally, never in git), judge/classifier models, web-search
  version, editable price table, and the credible-source allowlist.
- **Spending** — a persistent money-spent tracker: total, last 30 days, last 24 h, and a
  per-session history, tallied across every run.

Both front-ends ride the exact same core.

```
audio / transcript ─▶ STT ─▶ claim extraction ─▶ fact-check judge ─▶ verdict objects
                     (swappable)  (LLM classifier)   (API + web_search)   │
                                                                          ▼
                                                              live callbacks · .docx · .xlsx
```

## Verdicts

| Color | Meaning | Rule |
|-------|---------|------|
| 🟢 GREEN | Supported | ≥ `min_sources` (default 3) independent credible sources corroborate |
| 🔴 RED | Contradicted | Directly refuted **with a specific contradicting citation** |
| 🟡 YELLOW | Disputed / needs context | Partial, cherry-picked, out of date, or sources disagree |
| ⚪ GREY | Pending / unverifiable | Check running, evidence thin, or not a checkable claim (opinion/prediction/question) |

Calibration is enforced **twice** — the judge prompt biases toward YELLOW/GREY,
and `AnthropicJudge._calibrate()` re-checks in code: a RED with no contradicting
source is auto-downgraded to YELLOW, and a GREEN without enough independent
domains drops to YELLOW. The model can't hand back an over-confident verdict.

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Fastest path — replay a text transcript, no audio/whisper needed:
python run_headless.py --transcript samples/sample_debate.txt

# Real audio — transcribe a .wav locally, then fact-check:
#   pip install faster-whisper
python run_headless.py --wav debate.wav --stt faster_whisper
```

You'll see claims appear GREY ("checking…") and flip to their verdict color a few
seconds later as the async checks return. Exports land in `exports/`.

## Live desktop app

```bash
pip install -r requirements.txt        # fastapi/uvicorn/soundcard/faster-whisper/…
export ANTHROPIC_API_KEY=sk-ant-...     # or set it in the API tab once running
python run_gui.py                       # opens http://127.0.0.1:8760 in your browser
```

Pick a source — **System audio** (captures whatever is playing: a debate on TV, a
YouTube tab, any app, via WASAPI loopback), **Microphone** (in-person), or a
**.wav file** — set an optional per-session budget, and press Start. Claims stream
in GREY and flip color as each check returns; click one to expand its reasoning and
source links. A live **cost meter** tracks estimated API spend; set a budget above
0 and new checks stop once it's hit. "Save log" writes the `.docx` + `.xlsx` any time.

With an NVIDIA GPU, set Whisper **device = cuda** and **compute = float16** in the
Settings tab so live transcription keeps up. Settings, the API key, and spend history
are stored per-user in `%APPDATA%\GroundTruth` — never in the repo. The FastAPI backend
+ WebSocket event stream is the same component a future Chrome extension talks to.

## Build the installer (.exe)

```powershell
pip install -r requirements.txt pyinstaller
pyinstaller build\GroundTruth.spec --noconfirm        # -> dist\GroundTruth\GroundTruth.exe
```

`dist\GroundTruth\` is a self-contained folder — double-click `GroundTruth.exe` and the
app opens in your browser (no Python needed). To wrap it as a proper `Setup.exe` with
Start-Menu/desktop shortcuts, install [Inno Setup 6](https://jrsoftware.org/isdl.php) and:

```powershell
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\GroundTruth.iss
# -> installer\Output\GroundTruth-Setup.exe
```

Notes: it's a **one-folder** build (native Whisper/CTranslate2 DLLs are more reliable
unpacked than self-extracted). The bundle transcribes on **CPU** out of the box; GPU
inference needs the CUDA runtime libraries present on the machine. The packaged build
uses a built-in energy-based voice gate; installing `webrtcvad-wheels` in a source
checkout upgrades that to webrtcvad automatically.

## Layout

```
config.py                 # every knob: models, allowlist, thresholds, concurrency, audio, cost
run_headless.py           # Phase 0: reference wiring + terminal UI + export on exit
run_gui.py                # Phase 1: launches the live web UI
ddld/
  types.py                # Verdict / Utterance / Claim / Source / CheckedClaim
  pipeline.py             # orchestrator: emit PENDING now, check async, update in place
  cache.py                # normalize + dedup so repeated claims aren't re-checked
  audio/                  # AudioCapture ABC + soundcard loopback/mic backends  ← Phase 1
  stt/                    # STTEngine ABC + transcript_file + faster_whisper + streaming ← Phase 1
  claims/extractor.py     # heuristic prefilter + cheap LLM claim classifier
  factcheck/
    judge_prompt.py       # the strict, calibrated judge prompt (the crown jewel)
    anthropic_judge.py    # Messages API + web_search + code-side guardrails
  export/                 # docx + xlsx, color-coded, with disclaimer
server/                   # Phase 1 front-end
  app.py                  # FastAPI: runs Pipeline, streams events over WebSocket
  cost.py                 # live cost meter + budget-aware judge wrapper
  static/index.html       # the live color-coded feed
```

Every layer is behind an ABC (`STTEngine`, `Judge`), so a GUI or extension backend
consumes the same `Pipeline` callbacks and you can swap STT or the fact-check
backend without touching anything else.

## Decisions locked (with the forks left open)

| Decision | Locked default | Why / how to flip |
|----------|----------------|-------------------|
| Fact-check backend | Anthropic Messages API + `web_search` tool | You have an API account; one call does retrieval + grounded citations. Swap by implementing `Judge`. |
| Judge model | `claude-sonnet-5` | Cost/latency sweet spot per claim. If a model rejects web search, set `judge_model="claude-opus-4-8"`. |
| Classifier model | `claude-haiku-4-5` | Cheap pass to drop non-claims before spending a search. |
| Web search version | `web_search_20260318` | Current at build time; `allowed_domains` pins to the allowlist server-side. |
| Corroboration threshold | 3 independent sources for GREEN | `config.min_sources`. |
| STT (start) | `transcript_file` (test), `faster_whisper` (local audio) | Cloud streaming (Deepgram/AssemblyAI) drops in as another `STTEngine` for lower live latency. |
| Source allowlist | ~40 primary/official + fact-check + major outlet domains | Edit `DEFAULT_CREDIBLE_DOMAINS` in `config.py`. |

## Decisions locked for Phase 1

- **Front-end: local web UI (FastAPI + browser), not native or extension-first.** You want to cover every source — streaming, TV, in-person, files — and system-loopback capture from a desktop app is the superset that handles all of them. A local web backend also *is* the component a Chrome extension would talk to later, so it's built once, not thrown away.
- **Model strategy: Claude-only.** Claude's server-side `web_search` with domain allowlisting is the core feature; STT stays local (`faster-whisper`), private and free.
- **Cost control: live meter + optional per-session budget**, both in the UI. Estimated prices live in `config.model_prices` — edit to match current pricing.

## Roadmap

- **Phase 0 — Core:** transcript/wav → verdicts → export. ✅
- **Phase 1 — Live desktop app:** system-loopback / mic / file capture, streaming Whisper STT, live color feed with click-for-sources, cost meter + budget, save-log. ✅ *(PyInstaller `.exe` packaging still open — see below.)*
- **Phase 2 — Chrome extension:** MV3 `tabCapture`, reuse the same FastAPI backend + WebSocket contract, same UI language.
- **Phase 3 — Polish:** source-list config UI in the app, speaker diarization/labels, confidence tuning, latency batching + prioritizing high-salience claims.

## Still open

- **Package to a standalone `.exe`** with PyInstaller. Bundling `faster-whisper`/CTranslate2 CUDA libs into one file is finicky; worth doing once the app feels right.
- **Speaker diarization** ("who said what"). Single-stream transcript for now.

## Disclaimer

This is an automated, evidence-based assessment — not a definitive ruling. RED
means "contradicted by the cited evidence," not "lie." Always open the cited
sources and judge for yourself. The disclaimer ships in the UI and in every export.
