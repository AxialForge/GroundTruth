"""Judge interface. Swap in a different backend (local model + search API, etc.)
behind this ABC without touching the pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Claim, CheckedClaim


class Judge(ABC):
    @abstractmethod
    def check(self, claim: Claim) -> CheckedClaim:
        """Retrieve evidence and return a calibrated, cited verdict for one claim."""
        raise NotImplementedError
