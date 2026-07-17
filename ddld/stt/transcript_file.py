"""
Fake STT that replays a plain-text transcript.

This is the fastest way to exercise claim-extraction -> fact-check -> export
without downloading a Whisper model or wiring up audio. Use it first.

Line formats accepted (all optional prefixes):
    [01:23] Speaker Name: some spoken sentence.
    [83.0] Speaker: some spoken sentence.
    Speaker: some spoken sentence.
    just a bare line of speech.

Blank lines and lines starting with '#' are ignored.
"""
from __future__ import annotations

import re
import time
from typing import Iterator, Optional

from ..types import Utterance
from .base import STTEngine

_TS_BRACKET = re.compile(r"^\s*\[(?P<ts>[\d:.]+)\]\s*")
_SPEAKER = re.compile(r"^(?P<spk>[A-Z][\w .'-]{0,40}):\s+")


def _parse_ts(raw: str) -> Optional[float]:
    if ":" in raw:                      # mm:ss or hh:mm:ss
        parts = [float(p) for p in raw.split(":")]
        secs = 0.0
        for p in parts:
            secs = secs * 60 + p
        return secs
    try:
        return float(raw)               # bare seconds
    except ValueError:
        return None


class TranscriptFileSTT(STTEngine):
    def __init__(self, path: str, realtime: bool = True, default_gap: float = 3.0):
        self.path = path
        self.realtime = realtime        # pace yields to the transcript's own timestamps
        self.default_gap = default_gap  # seconds per line when no timestamps are present

    def stream(self) -> Iterator[Utterance]:
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh]

        clock = 0.0
        prev_emit = 0.0
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            speaker: Optional[str] = None
            start = clock

            m = _TS_BRACKET.match(line)
            if m:
                ts = _parse_ts(m.group("ts"))
                if ts is not None:
                    start = ts
                line = line[m.end():]

            m = _SPEAKER.match(line)
            if m:
                speaker = m.group("spk").strip()
                line = line[m.end():]

            text = line.strip()
            if not text:
                continue

            end = start + self.default_gap
            clock = end

            if self.realtime:
                # sleep the gap between this utterance and the previous emit,
                # so downstream sees speech arrive at a lifelike pace
                wait = max(0.0, start - prev_emit)
                time.sleep(min(wait, self.default_gap))  # cap so demos don't stall
                prev_emit = start

            yield Utterance(text=text, start=start, end=end, speaker=speaker)
