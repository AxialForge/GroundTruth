"""
Fact-check judge backed by the Anthropic Messages API + built-in web_search tool.

One create() call does retrieval AND grounded assessment. `allowed_domains` pins
searches to the credible-source allowlist server-side, so the judge can only cite
from domains you approved.

Two safety layers:
  1. The prompt (judge_prompt.py) enforces the calibration philosophy.
  2. Code here re-checks the verdict against the returned sources and DOWNGRADES
     anything unsupported — a RED with no contradicting citation becomes YELLOW,
     a GREEN without enough independent sources becomes YELLOW. Defense in depth:
     the model can't hand back an over-confident verdict even if it tries.
"""
from __future__ import annotations

import json
import time
from urllib.parse import urlparse

from ..types import Claim, CheckedClaim, Source, Verdict
from .base import Judge
from .judge_prompt import build_system, build_user


def dedupe_domains(domains) -> list:
    """Normalize + de-duplicate an allowlist. The web_search tool rejects
    `allowed_domains` containing duplicates (case-insensitively), and a
    user-edited list is easy to get wrong — so clean it at every call site."""
    seen, out = set(), []
    for d in domains or []:
        norm = (d or "").strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


class AnthropicJudge(Judge):
    def __init__(self, client, cfg):
        self._client = client
        self._cfg = cfg
        self._system = build_system(cfg.min_sources)

    def _web_search_tool(self) -> dict:
        tool = {
            "type": self._cfg.web_search_tool_version,
            "name": "web_search",
            "max_uses": self._cfg.max_search_uses,
        }
        domains = dedupe_domains(self._cfg.credible_domains)
        if domains:
            # Only one of allowed_domains / blocked_domains may be set, and it
            # must not contain duplicates.
            tool["allowed_domains"] = domains
        return tool

    def check(self, claim: Claim) -> CheckedClaim:
        result = CheckedClaim(claim=claim)
        if not claim.checkable:
            result.verdict = Verdict.PENDING
            result.reasoning = "Not a checkable factual claim (opinion / prediction / question)."
            result.checked_at = time.time()
            return result

        try:
            raw = self._run(claim)
            self._apply(result, raw)
        except Exception as e:  # keep the pipeline alive; surface the error on the item
            result.verdict = Verdict.PENDING
            result.error = str(e)
            result.reasoning = "Check failed; left unverified."
        result.checked_at = time.time()
        return result

    # ------------------------------------------------------------------ #
    def _run(self, claim: Claim) -> dict:
        messages = [{"role": "user", "content": build_user(claim.text, claim.speaker)}]
        tools = [self._web_search_tool()]

        resp = self._client.messages.create(
            model=self._cfg.judge_model,
            max_tokens=2000,
            system=self._system,
            tools=tools,
            messages=messages,
        )
        # Web search can pause a long turn; resume until it settles.
        guard = 0
        while resp.stop_reason == "pause_turn" and guard < 4:
            messages.append({"role": "assistant", "content": resp.content})
            resp = self._client.messages.create(
                model=self._cfg.judge_model,
                max_tokens=2000,
                system=self._system,
                tools=tools,
                messages=messages,
            )
            guard += 1

        text = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        )
        return _loads_json_object(text)

    def _apply(self, result: CheckedClaim, raw: dict) -> None:
        sources = []
        for s in raw.get("sources", []) or []:
            url = (s.get("url") or "").strip()
            if not url:
                continue
            sources.append(
                Source(
                    url=url,
                    title=(s.get("title") or "").strip(),
                    quote=(s.get("quote") or "").strip(),
                    stance=(s.get("stance") or "context").strip().lower(),
                )
            )
        result.sources = sources
        result.reasoning = (raw.get("reasoning") or "").strip()
        try:
            result.confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            result.confidence = 0.0

        verdict = _to_verdict(raw.get("verdict"))
        result.verdict = self._calibrate(verdict, sources)

    def _calibrate(self, verdict: Verdict, sources: list[Source]) -> Verdict:
        """Code-side guardrails. The model's verdict only stands if the evidence backs it."""
        if verdict is Verdict.CONTRADICTED:
            # RED requires at least one explicitly contradicting citation.
            if not any(s.stance == "contradicts" for s in sources):
                return Verdict.DISPUTED
            return Verdict.CONTRADICTED

        if verdict is Verdict.SUPPORTED:
            supporting_domains = {
                _domain(s.url) for s in sources if s.stance == "supports"
            }
            if len(supporting_domains) < self._cfg.min_sources:
                return Verdict.DISPUTED
            return Verdict.SUPPORTED

        return verdict  # YELLOW / GREY pass through


# ---------------------------------------------------------------------- #
def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or url).lower().removeprefix("www.")
    except Exception:
        return url.lower()


def _to_verdict(value) -> Verdict:
    v = (str(value) or "").strip().upper()
    for member in Verdict:
        if member.value == v or member.name == v:
            return member
    return Verdict.PENDING


def _loads_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        brace = text.find("{")
        if brace != -1:
            text = text[brace:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge did not return a JSON object: {text[:200]!r}")
    return json.loads(text[start:end + 1])
