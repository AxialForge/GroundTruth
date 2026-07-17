"""
STT engine interface. Everything the pipeline needs from speech-to-text is
`stream()` yielding finalized Utterances. Swap cloud/local behind this ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..types import Utterance


class STTEngine(ABC):
    @abstractmethod
    def stream(self) -> Iterator[Utterance]:
        """Yield Utterance objects as they are finalized, in time order.

        Implementations block between yields as needed (e.g. pacing to real time,
        or waiting on a model). The pipeline consumes this on a background thread.
        """
        raise NotImplementedError
