"""
Audio capture interface (Phase 1).

A capture backend yields short blocks of float32 mono PCM in [-1, 1] at a fixed
sample rate. It knows nothing about speech, claims, or Whisper — the streaming
STT engine downstream re-frames these blocks for VAD and transcription.

Contract:
  frames() -> Iterator[np.ndarray]   # 1-D float32, [-1, 1], `sample_rate` Hz
  stop()                             # ask frames() to finish (call from another thread)

Backends block inside frames() waiting on the sound card; the pipeline consumes
this on a background thread, exactly like every other STTEngine source.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class AudioCapture(ABC):
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._stop = threading.Event()

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield float32 mono blocks until stop() is called or the source ends."""
        raise NotImplementedError

    def stop(self) -> None:
        """Signal frames() to stop. Safe to call from any thread."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()
