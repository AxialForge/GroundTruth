"""
Live cost meter + budget ceiling.

Non-invasive: the core (ClaimExtractor, AnthropicJudge) still just calls
`client.messages.create(...)`. We hand them a thin proxy that tallies token +
web_search usage off each response into a shared CostMeter, and wrap the judge
so it stops spending once a per-session budget is hit.

Prices come from config.model_prices (USD per 1M tokens) and
config.web_search_price_per_1k. They're ESTIMATES — see config.py.
"""
from __future__ import annotations

import threading

from ddld.factcheck.base import Judge
from ddld.types import Claim, CheckedClaim, Verdict


class CostMeter:
    """Thread-safe running tally across every API call in a session."""

    def __init__(self, model_prices: dict, web_search_price_per_1k: float):
        self._prices = model_prices
        self._search_price = web_search_price_per_1k
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.web_searches = 0
        self.api_calls = 0
        self.cost_usd = 0.0

    def record(self, model: str, usage) -> None:
        if usage is None:
            return
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        # cached input tokens are billed differently; fold them into input as an
        # over-estimate rather than under-count (the meter should never flatter you).
        in_tok += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        in_tok += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

        searches = 0
        server_tool = getattr(usage, "server_tool_use", None)
        if server_tool is not None:
            searches = int(getattr(server_tool, "web_search_requests", 0) or 0)

        price = self._prices.get(model, {})
        delta = (
            in_tok / 1_000_000 * float(price.get("in", 0.0))
            + out_tok / 1_000_000 * float(price.get("out", 0.0))
            + searches / 1000 * float(self._search_price)
        )
        with self._lock:
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.web_searches += searches
            self.api_calls += 1
            self.cost_usd += delta

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cost_usd": round(self.cost_usd, 4),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "web_searches": self.web_searches,
                "api_calls": self.api_calls,
            }


class _MeteredMessages:
    def __init__(self, real, meter: CostMeter):
        self._real = real
        self._meter = meter

    def create(self, **kwargs):
        resp = self._real.create(**kwargs)
        try:
            self._meter.record(kwargs.get("model", ""), getattr(resp, "usage", None))
        except Exception:  # metering must never break a check
            pass
        return resp


class MeteredAnthropic:
    """Proxy that looks like an Anthropic client but tallies usage on `.messages.create`."""

    def __init__(self, client, meter: CostMeter):
        self._client = client
        self._messages = _MeteredMessages(client.messages, meter)

    @property
    def messages(self):
        return self._messages

    def __getattr__(self, name):
        return getattr(self._client, name)


class BudgetedJudge(Judge):
    """Wraps a Judge; short-circuits to PENDING once the session budget is spent."""

    def __init__(self, inner: Judge, meter: CostMeter, budget_usd: float):
        self._inner = inner
        self._meter = meter
        self._budget = budget_usd

    def check(self, claim: Claim) -> CheckedClaim:
        if self._budget and self._meter.cost_usd >= self._budget:
            result = CheckedClaim(claim=claim)
            result.verdict = Verdict.PENDING
            result.reasoning = (
                f"Skipped — session budget (${self._budget:.2f}) reached. "
                "Raise it in the UI to keep checking."
            )
            return result
        return self._inner.check(claim)
