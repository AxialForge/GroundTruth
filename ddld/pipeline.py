"""
The front-end-agnostic core. Wire audio/STT + extractor + judge into it, subscribe
callbacks, and drive it. A terminal, a desktop GUI, or a Chrome extension backend
all consume the exact same events.

Event model (the key UX contract from the spec):
  on_claim(checked)   -> fired IMMEDIATELY when a claim appears, in PENDING (GREY) state.
  on_verdict(checked) -> fired later, in place, when the async check returns.

Verification lags speech by design. This orchestrator embraces that: it never
blocks the transcript on a check.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .cache import ClaimCache
from .claims import ClaimExtractor
from .factcheck.base import Judge
from .stt.base import STTEngine
from .types import CheckedClaim, Utterance, Verdict

OnClaim = Callable[[CheckedClaim], None]
OnVerdict = Callable[[CheckedClaim], None]
OnUtterance = Callable[[Utterance], None]


class Pipeline:
    def __init__(
        self,
        stt: STTEngine,
        extractor: ClaimExtractor,
        judge: Judge,
        cache: Optional[ClaimCache] = None,
        max_workers: int = 4,
        extract_batch: int = 3,
    ):
        self._stt = stt
        self._extractor = extractor
        self._judge = judge
        self._cache = cache
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._extract_batch = extract_batch

        self.on_utterance: Optional[OnUtterance] = None
        self.on_claim: Optional[OnClaim] = None
        self.on_verdict: Optional[OnVerdict] = None

        self._checked: list[CheckedClaim] = []
        self._transcript: list[Utterance] = []
        self._lock = threading.Lock()
        self._pending_futures: list = []

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Blocking: consume the STT stream to completion, then drain checks."""
        batch: list[Utterance] = []
        for utt in self._stt.stream():
            with self._lock:
                self._transcript.append(utt)
            if self.on_utterance:
                self.on_utterance(utt)

            batch.append(utt)
            if len(batch) >= self._extract_batch:
                self._process_batch(batch)
                batch = []

        if batch:
            self._process_batch(batch)

        # Let in-flight checks finish.
        for fut in list(self._pending_futures):
            fut.result()
        self._pool.shutdown(wait=True)

    def _process_batch(self, batch: list[Utterance]) -> None:
        for claim in self._extractor.extract(batch):
            checked = CheckedClaim(claim=claim)
            if not claim.checkable:
                checked.verdict = Verdict.PENDING
                checked.reasoning = "Not a checkable factual claim."
                self._record(checked)
                if self.on_claim:
                    self.on_claim(checked)
                continue

            # Emit PENDING immediately (GREY on screen)...
            self._record(checked)
            if self.on_claim:
                self.on_claim(checked)

            # Cache hit? resolve without spending a search.
            if self._cache:
                hit = self._cache.get(claim)
                if hit is not None:
                    self._resolve(checked, hit)
                    continue

            # ...then check asynchronously and update in place.
            fut = self._pool.submit(self._judge.check, claim)
            fut.add_done_callback(lambda f, c=checked: self._on_done(f, c))
            self._pending_futures.append(fut)

    def _on_done(self, future, placeholder: CheckedClaim) -> None:
        try:
            result = future.result()
        except Exception as e:  # pragma: no cover
            placeholder.error = str(e)
            placeholder.verdict = Verdict.PENDING
            if self.on_verdict:
                self.on_verdict(placeholder)
            return
        if self._cache:
            self._cache.put(result)
        self._resolve(placeholder, result)

    def _resolve(self, placeholder: CheckedClaim, result: CheckedClaim) -> None:
        """Copy the verdict onto the already-emitted placeholder and notify."""
        placeholder.verdict = result.verdict
        placeholder.confidence = result.confidence
        placeholder.reasoning = result.reasoning
        placeholder.sources = result.sources
        placeholder.checked_at = result.checked_at
        placeholder.error = result.error
        if self.on_verdict:
            self.on_verdict(placeholder)

    def _record(self, checked: CheckedClaim) -> None:
        with self._lock:
            self._checked.append(checked)

    # ------------------------------------------------------------------ #
    @property
    def transcript(self) -> list[Utterance]:
        with self._lock:
            return list(self._transcript)

    @property
    def checked_claims(self) -> list[CheckedClaim]:
        with self._lock:
            return list(self._checked)
