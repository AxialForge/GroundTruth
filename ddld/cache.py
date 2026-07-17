"""
Claim dedup cache. Debates repeat talking points; without this you pay to check
the same claim five times. Normalizes claim text to a key, persists to JSON.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Optional

from .types import CheckedClaim, Claim, Source, Verdict

_NORMALIZE = re.compile(r"[^a-z0-9 ]+")


def normalize_key(text: str) -> str:
    t = text.lower().strip()
    t = _NORMALIZE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


class ClaimCache:
    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._store: dict[str, dict] = {}
        if enabled and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._store = json.load(fh)
            except Exception:
                self._store = {}

    def get(self, claim: Claim) -> Optional[CheckedClaim]:
        if not self.enabled:
            return None
        row = self._store.get(normalize_key(claim.text))
        if not row:
            return None
        return _rehydrate(claim, row)

    def put(self, checked: CheckedClaim) -> None:
        if not self.enabled:
            return
        # Don't cache failures or still-pending items.
        if checked.error or checked.verdict is Verdict.PENDING:
            return
        self._store[normalize_key(checked.claim.text)] = {
            "verdict": checked.verdict.value,
            "confidence": checked.confidence,
            "reasoning": checked.reasoning,
            "sources": [asdict(s) for s in checked.sources],
        }

    def flush(self) -> None:
        if not self.enabled:
            return
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._store, fh, ensure_ascii=False, indent=2)


def _rehydrate(claim: Claim, row: dict) -> CheckedClaim:
    return CheckedClaim(
        claim=claim,
        verdict=Verdict(row["verdict"]),
        confidence=row.get("confidence", 0.0),
        reasoning=row.get("reasoning", ""),
        sources=[Source(**s) for s in row.get("sources", [])],
    )
