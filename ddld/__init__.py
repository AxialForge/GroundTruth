"""
Debate Lie Detector — front-end-agnostic core.

    audio/transcript -> STT -> claim extraction -> fact-check judge -> verdict objects

Build any front-end (terminal, desktop, Chrome extension) on top of Pipeline's
callbacks. See run_headless.py for the reference wiring.
"""
from .types import Utterance, Claim, CheckedClaim, Source, Verdict
from .pipeline import Pipeline
from .claims import ClaimExtractor
from .factcheck import AnthropicJudge, Judge
from .cache import ClaimCache
from .stt import make_stt, STTEngine

__all__ = [
    "Utterance", "Claim", "CheckedClaim", "Source", "Verdict",
    "Pipeline", "ClaimExtractor", "AnthropicJudge", "Judge",
    "ClaimCache", "make_stt", "STTEngine",
]
