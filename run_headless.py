#!/usr/bin/env python3
"""
Phase 0 headless runner. Proves the whole pipeline in a terminal.

Examples
--------
  # Fastest: replay a text transcript (no audio, no whisper), fact-check, export
  python run_headless.py --transcript samples/sample_debate.txt

  # Real audio: transcribe a .wav locally with faster-whisper, then fact-check
  python run_headless.py --wav debate.wav --stt faster_whisper

Needs ANTHROPIC_API_KEY in the environment for claim extraction + fact-checking.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import anthropic

from config import config
from ddld import AnthropicJudge, ClaimCache, ClaimExtractor, Pipeline, Verdict, make_stt
from ddld.export import export_docx, export_xlsx

# ANSI colors for the live feed.
_ANSI = {
    Verdict.SUPPORTED: "\033[42m\033[30m",   # green bg
    Verdict.CONTRADICTED: "\033[41m\033[37m",  # red bg
    Verdict.DISPUTED: "\033[43m\033[30m",    # yellow bg
    Verdict.PENDING: "\033[100m\033[37m",    # grey bg
}
_RESET = "\033[0m"


def _badge(v: Verdict) -> str:
    return f"{_ANSI[v]} {v.label.upper()} {_RESET}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Debate Lie Detector — headless")
    ap.add_argument("--transcript", help="path to a .txt transcript (transcript_file mode)")
    ap.add_argument("--wav", help="path to a .wav file (faster_whisper mode)")
    ap.add_argument("--stt", choices=["transcript_file", "faster_whisper"],
                    help="override config.stt_engine")
    ap.add_argument("--no-realtime", action="store_true",
                    help="don't pace the transcript to its timestamps (run flat out)")
    ap.add_argument("--out", default=None, help="export basename (default: timestamped)")
    args = ap.parse_args()

    if args.stt:
        config.stt_engine = args.stt
    if args.no_realtime:
        config.realtime_playback = False

    source = args.transcript or args.wav
    if not source:
        ap.error("provide --transcript <file.txt> or --wav <file.wav>")
    if args.wav and not args.stt:
        config.stt_engine = "faster_whisper"

    try:
        config.validate()
    except RuntimeError as e:
        print(f"[config] {e}", file=sys.stderr)
        return 2

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    stt = make_stt(config, source)
    extractor = ClaimExtractor(client, config.classifier_model)
    judge = AnthropicJudge(client, config)
    cache = ClaimCache(config.cache_path, enabled=config.enable_cache)

    pipe = Pipeline(
        stt=stt, extractor=extractor, judge=judge, cache=cache,
        max_workers=config.max_concurrent_checks,
    )

    print("=" * 70)
    print("  DEBATE LIE DETECTOR  ·  headless")
    print(f"  {config.DISCLAIMER}")
    print("=" * 70)

    def on_utterance(u):
        who = f"{u.speaker}: " if u.speaker else ""
        print(f"\n\033[2m[{u.start:5.0f}s]\033[0m {who}{u.text}")

    def on_claim(c):
        if c.claim.checkable:
            print(f"    {_badge(Verdict.PENDING)} checking… \033[2m{c.claim.text}\033[0m")

    def on_verdict(c):
        if c.error:
            print(f"    {_badge(Verdict.PENDING)} (error) {c.claim.text}  \033[2m{c.error}\033[0m")
            return
        if not c.claim.checkable:
            return
        conf = f"{c.confidence:.2f}"
        print(f"    {_badge(c.verdict)} conf={conf}  {c.claim.text}")
        if c.reasoning:
            print(f"        \033[2m↳ {c.reasoning}\033[0m")
        for s in c.sources[:3]:
            print(f"        \033[2m· [{s.stance}] {s.url}\033[0m")

    pipe.on_utterance = on_utterance
    pipe.on_claim = on_claim
    pipe.on_verdict = on_verdict

    t0 = time.time()
    pipe.run()
    cache.flush()

    # Exports.
    os.makedirs(config.export_dir, exist_ok=True)
    base = args.out or time.strftime("session_%Y%m%d_%H%M%S")
    docx_path = os.path.join(config.export_dir, base + ".docx")
    xlsx_path = os.path.join(config.export_dir, base + ".xlsx")
    export_docx(docx_path, pipe.transcript, pipe.checked_claims, config.DISCLAIMER)
    export_xlsx(xlsx_path, pipe.transcript, pipe.checked_claims, config.DISCLAIMER)

    checkable = [c for c in pipe.checked_claims if c.claim.checkable]
    print("\n" + "=" * 70)
    print(f"  Done in {time.time() - t0:.1f}s · {len(checkable)} claims checked")
    print(f"  Word : {docx_path}")
    print(f"  Excel: {xlsx_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
