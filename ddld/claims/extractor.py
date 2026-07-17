"""
Claim extraction (pipeline step 3).

Two stages, cheapest first:
  1. Heuristic prefilter — drop the obvious non-claims (empty, too short, bare
     questions) for free, before spending any tokens.
  2. LLM classifier (cheap model) — for what survives, decide checkable vs. not,
     and split a sentence into one or more *atomic* factual claims. Atomizing
     matters: "Unemployment fell to 4% and inflation is the lowest in 30 years"
     is two independently checkable claims.

Non-claims (opinions, predictions, jokes, questions, pure rhetoric) are returned
with checkable=False so the pipeline can render them GREY and never spend a search.
"""
from __future__ import annotations

import json
from typing import Optional

from ..types import Claim, Utterance

_MIN_WORDS = 4

_SYSTEM = """You separate checkable factual claims from everything else in debate speech.

A CHECKABLE FACTUAL CLAIM asserts something about the world that could, in principle,
be verified or refuted against evidence: statistics, historical events, records of who
did/said/voted what, quantities, dates, scientific facts, attributable actions.

NOT checkable (checkable=false): opinions, value judgements, predictions about the
future, hypotheticals, rhetorical questions, jokes, vague generalities, and pure
promises ("I will fix this").

For each input utterance, split it into ATOMIC claims — one verifiable assertion each.
Normalize each claim into a clear, self-contained sentence (resolve pronouns using the
utterance's own context; do not invent facts not present).

Return ONLY a JSON array, one object per input utterance, in order:
[{"i": <index>, "checkable": <bool>, "claims": ["<atomic claim>", ...]}]
If checkable is false, "claims" must be []. No prose, no markdown fences."""


def _prefilter(text: str) -> bool:
    """True if worth sending to the classifier."""
    t = text.strip()
    if len(t.split()) < _MIN_WORDS:
        return False
    # A lone question is not a claim. (Statements containing a '?' mid-way still pass.)
    if t.endswith("?") and t.count(".") == 0:
        return False
    return True


class ClaimExtractor:
    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def extract(self, utterances: list[Utterance]) -> list[Claim]:
        """Batch-classify utterances and return the atomic claims (checkable + non)."""
        candidates = [(i, u) for i, u in enumerate(utterances) if _prefilter(u.text)]
        non_claims = [u for i, u in enumerate(utterances) if not _prefilter(u.text)]

        results: list[Claim] = []
        # Non-claims still get surfaced (GREY), so the transcript stays complete.
        for u in non_claims:
            results.append(
                Claim(text=u.text, utterance_id=u.id, speaker=u.speaker,
                      t_start=u.start, checkable=False)
            )

        if not candidates:
            return results

        payload = [{"i": i, "text": u.text} for i, u in candidates]
        try:
            parsed = self._classify(payload)
        except Exception:
            # Fail open: if the classifier errors, treat candidates as checkable
            # single claims rather than dropping them silently.
            for i, u in candidates:
                results.append(
                    Claim(text=u.text, utterance_id=u.id, speaker=u.speaker,
                          t_start=u.start, checkable=True)
                )
            return results

        by_index = {c[0]: c[1] for c in candidates}
        for row in parsed:
            i = row.get("i")
            u = by_index.get(i)
            if u is None:
                continue
            if row.get("checkable") and row.get("claims"):
                for atom in row["claims"]:
                    atom = (atom or "").strip()
                    if atom:
                        results.append(
                            Claim(text=atom, utterance_id=u.id, speaker=u.speaker,
                                  t_start=u.start, checkable=True)
                        )
            else:
                results.append(
                    Claim(text=u.text, utterance_id=u.id, speaker=u.speaker,
                          t_start=u.start, checkable=False)
                )
        return results

    def _classify(self, payload: list[dict]) -> list[dict]:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1500,
            system=_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _loads_json_array(text)


def _loads_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("["):]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("classifier did not return a JSON array")
    return json.loads(text[start:end + 1])
