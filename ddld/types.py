"""
Core data model for the Debate Lie Detector.

Everything upstream (audio, STT, claim extraction) and downstream (rendering,
export) speaks in these objects, so front-ends stay decoupled from the pipeline.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    """The four on-screen states. Values are the display colors."""

    SUPPORTED = "GREEN"       # corroborated by >= MIN_SOURCES independent credible sources
    CONTRADICTED = "RED"      # directly refuted; REQUIRES a contradicting citation
    DISPUTED = "YELLOW"       # partial / cherry-picked / sources disagree / missing context
    PENDING = "GREY"          # check still running, OR not a checkable factual claim

    @property
    def label(self) -> str:
        return {
            "GREEN": "Supported",
            "RED": "Contradicted",
            "YELLOW": "Disputed / Needs context",
            "GREY": "Pending / Unverifiable",
        }[self.value]

    @property
    def hex(self) -> str:
        """Fill color for exports / UI."""
        return {
            "GREEN": "C6EFCE",
            "RED": "FFC7CE",
            "YELLOW": "FFEB9C",
            "GREY": "D9D9D9",
        }[self.value]


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Utterance:
    """A finalized chunk of transcript with timing (and optional speaker)."""

    text: str
    start: float                      # seconds from stream start
    end: float
    speaker: Optional[str] = None
    id: str = field(default_factory=_short_id)


@dataclass
class Source:
    """One piece of evidence the judge used."""

    url: str
    title: str = ""
    quote: str = ""                   # keep short (<= ~15 words); attribution, not reproduction
    stance: str = "context"           # "supports" | "contradicts" | "context"


@dataclass
class Claim:
    """A single, normalized, atomic factual assertion pulled from an utterance."""

    text: str
    utterance_id: str
    speaker: Optional[str] = None
    t_start: float = 0.0
    checkable: bool = True            # False => opinion / prediction / question / rhetoric
    id: str = field(default_factory=_short_id)


@dataclass
class CheckedClaim:
    """A claim plus its verdict. Starts PENDING, updated in place when the check returns."""

    claim: Claim
    verdict: Verdict = Verdict.PENDING
    confidence: float = 0.0           # 0..1, judge's calibrated confidence
    reasoning: str = ""               # one-line rationale
    sources: list[Source] = field(default_factory=list)
    checked_at: Optional[float] = None
    error: Optional[str] = None       # populated if the check failed

    # ------- helpers for rendering / export -------
    @property
    def supporting(self) -> list[Source]:
        return [s for s in self.sources if s.stance == "supports"]

    @property
    def contradicting(self) -> list[Source]:
        return [s for s in self.sources if s.stance == "contradicts"]

    def as_row(self) -> dict:
        """Flat dict for xlsx/docx tables."""
        return {
            "timestamp_s": round(self.claim.t_start, 1),
            "speaker": self.claim.speaker or "",
            "claim": self.claim.text,
            "verdict": self.verdict.value,
            "verdict_label": self.verdict.label,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "sources": " | ".join(s.url for s in self.sources),
            "checked_at": self.checked_at or "",
        }
