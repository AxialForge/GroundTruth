"""
Central configuration. One place for every tunable knob.

Nothing else in the codebase should hard-code a model name, a domain, or a
threshold — it all comes from here so the whole tool re-tunes from one file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ------------------------------------------------------------------ #
#  Default credible-source allowlist.
#  Prefer primary/official records + data over secondary reporting.
#  This is enforced SERVER-SIDE via the web_search tool's allowed_domains,
#  so the judge can only cite from this list. Edit freely.
# ------------------------------------------------------------------ #
DEFAULT_CREDIBLE_DOMAINS: list[str] = [
    # Primary / official records and data
    "gao.gov", "cbo.gov", "gpo.gov", "congress.gov", "govinfo.gov",
    "bls.gov", "census.gov", "bea.gov", "federalreserve.gov", "fred.stlouisfed.org",
    "cdc.gov", "nih.gov", "fda.gov", "epa.gov", "noaa.gov", "nasa.gov",
    "supremecourt.gov", "justice.gov", "treasury.gov", "state.gov",
    "who.int", "un.org", "worldbank.org", "imf.org", "oecd.org", "eurostat.ec.europa.eu",
    # Established fact-check orgs
    "factcheck.org", "politifact.com", "apnews.com", "fullfact.org",
    # Established outlets (secondary — used to corroborate, not as primary source of truth)
    "reuters.com", "bbc.com", "npr.org", "pbs.org",
    "nytimes.com", "washingtonpost.com", "wsj.com", "economist.com",
    "nature.com", "science.org", "pnas.org", "thelancet.com", "nejm.org",
]


# ------------------------------------------------------------------ #
#  Rough API price table for the live cost meter (USD per 1M tokens).
#  ESTIMATES — verify against current Anthropic pricing and edit freely.
#  The meter is a running approximation to keep a session from surprising
#  you, not an invoice. web_search is billed per request.
# ------------------------------------------------------------------ #
DEFAULT_MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5":            {"in": 3.00,  "out": 15.00},
    "claude-opus-4-8":            {"in": 15.00, "out": 75.00},
    "claude-haiku-4-5-20251001":  {"in": 0.80,  "out": 4.00},
}
DEFAULT_WEB_SEARCH_PRICE_PER_1K: float = 10.00   # USD per 1000 web_search requests


@dataclass
class Config:
    # --- Anthropic API (the fact-check backend) ---
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))

    # Judge = grounded assessment with citations. Web search needs a compatible model.
    # sonnet-5 is the cost/latency sweet spot for per-claim volume.
    # If you hit a "model does not support web_search" error, switch to "claude-opus-4-8".
    judge_model: str = "claude-sonnet-5"

    # Classifier = cheap pass that separates checkable claims from opinion/rhetoric/questions.
    classifier_model: str = "claude-haiku-4-5-20251001"

    # Verified current as of build time. Bump when Anthropic ships a newer version.
    web_search_tool_version: str = "web_search_20260318"
    max_search_uses: int = 5              # per claim, hard cap on searches

    # --- Verdict calibration ---
    min_sources: int = 3                  # independent credible sources needed for GREEN
    # Bias toward YELLOW/GREY. A false RED/GREEN is worse than an honest "not sure".

    # --- Source policy ---
    credible_domains: list[str] = field(default_factory=lambda: list(DEFAULT_CREDIBLE_DOMAINS))

    # --- Concurrency / cost control ---
    max_concurrent_checks: int = 4        # parallel judge calls in flight
    cache_path: str = "claim_cache.json"  # dedup so repeated claims aren't re-checked
    enable_cache: bool = True

    # --- STT ---
    # "transcript_file"  -> no audio, replays a text transcript (fastest way to test the pipeline)
    # "faster_whisper"   -> local model, transcribes a finished .wav file (private, no per-minute cost)
    # "streaming"        -> LIVE: capture mic / system-loopback audio, VAD-segment, transcribe in near-real-time
    stt_engine: str = "transcript_file"
    whisper_model_size: str = "small"     # tiny|base|small|medium|large-v3
    whisper_device: str = "auto"          # auto|cpu|cuda   (with an NVIDIA GPU, "cuda" keeps up live)
    whisper_compute_type: str = "auto"    # auto|float16|int8_float16|int8  (float16 on CUDA is a good default)
    realtime_playback: bool = True        # in transcript_file mode, pace utterances to their timestamps

    # --- Live audio capture (streaming STT) ---
    # audio_source: "loopback" (whatever is playing on this PC — TV, browser, app),
    #               "mic" (microphone, for in-person), or "file" (a .wav, non-live).
    audio_source: str = "loopback"
    audio_device_id: str = ""             # "" = system default for the chosen source
    sample_rate: int = 16000              # whisper + webrtcvad both want 16 kHz mono
    vad_aggressiveness: int = 2           # 0..3, webrtcvad; higher = more eager to call audio "not speech"
    vad_silence_ms: int = 700             # trailing silence that finalizes an utterance
    vad_min_segment_ms: int = 400         # drop blips shorter than this (coughs, clicks)
    vad_max_segment_s: float = 14.0       # force-flush a long monologue so verdicts don't stall

    # --- Local web UI (server front-end) ---
    server_host: str = "127.0.0.1"        # localhost only; this is a personal tool, don't expose it
    server_port: int = 8760

    # --- Live cost meter / budget ---
    model_prices: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_MODEL_PRICES.items()})
    web_search_price_per_1k: float = DEFAULT_WEB_SEARCH_PRICE_PER_1K
    session_budget_usd: float = 0.0       # 0 = unlimited. Above 0, new checks are skipped once the meter hits it.

    # --- Export ---
    export_dir: str = "exports"

    DISCLAIMER: str = (
        "This is an automated, evidence-based assessment generated by matching spoken "
        "claims against cited web sources — not a definitive ruling. RED means "
        "'contradicted by the cited evidence,' not 'lie.' Always open the cited sources "
        "and judge for yourself."
    )

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or put it in a .env / your shell profile."
            )


# A ready-to-use default instance.
config = Config()
