"""
STT engine interface. Everything the pipeline needs from speech-to-text is
`stream()` yielding finalized Utterances. Swap cloud/local behind this ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..types import Utterance


# Per-process memo of what device actually worked, so repeated Starts don't
# re-probe a GPU we already know can't run inference (and don't load the model
# twice each time, which widens the cache race below).
_RESOLVED_DEVICE: dict = {}


def load_whisper_model(WhisperModel, model_size: str, device: str, compute_type: str):
    """Build a faster-whisper model on the requested device, robustly.

    Two failure modes handled here:
      * GPU can't run inference — WhisperModel(device='auto'/'cuda') *loads* fine
        on a GPU machine, but the first transcribe needs cuBLAS (cublas64_12.dll),
        absent unless a CUDA toolkit is installed. We PROBE with a 1s silent
        transcribe to force the CUDA libs now; on failure we fall back to CPU.
      * Transient 'Unable to open file model.bin' — HuggingFace re-links the
        snapshot on load, and CTranslate2 can catch model.bin mid-relink. We
        retry a few times with a short pause."""
    import time
    import numpy as np

    def _build(dev: str, ct: str):
        model = WhisperModel(model_size, device=dev, compute_type=ct)
        segments, _info = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
        for _ in segments:
            break
        return model

    def _build_retry(dev: str, ct: str, attempts: int = 4):
        for i in range(attempts):
            try:
                return _build(dev, ct)
            except Exception as e:
                transient = "model.bin" in str(e) or "Unable to open file" in str(e)
                if transient and i < attempts - 1:
                    print(f"[stt] model files busy (HF cache relink); retry {i + 1}…")
                    time.sleep(0.8)
                    continue
                raise

    key = (model_size, (device or "").lower(), (compute_type or "").lower())
    resolved = _RESOLVED_DEVICE.get(key)
    if resolved:
        return _build_retry(*resolved)

    try:
        model = _build_retry(device, compute_type)
        _RESOLVED_DEVICE[key] = (device, compute_type)
        return model
    except Exception as e:
        if (device or "").lower() == "cpu":
            raise
        print(f"[stt] Whisper on device={device!r} can't run ({type(e).__name__}: {e}); "
              f"falling back to CPU. (Set device=cpu in Settings to skip this probe.)")
        model = _build_retry("cpu", "int8")
        _RESOLVED_DEVICE[key] = ("cpu", "int8")
        return model


class STTEngine(ABC):
    @abstractmethod
    def stream(self) -> Iterator[Utterance]:
        """Yield Utterance objects as they are finalized, in time order.

        Implementations block between yields as needed (e.g. pacing to real time,
        or waiting on a model). The pipeline consumes this on a background thread.
        """
        raise NotImplementedError
