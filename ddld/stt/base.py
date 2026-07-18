"""
STT engine interface. Everything the pipeline needs from speech-to-text is
`stream()` yielding finalized Utterances. Swap cloud/local behind this ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..types import Utterance


def load_whisper_model(WhisperModel, model_size: str, device: str, compute_type: str):
    """Build a faster-whisper model on the requested device, falling back to CPU
    if the GPU can't actually run inference.

    Key subtlety: on a GPU machine, WhisperModel(device='auto'/'cuda') *loads*
    fine, but the first transcribe needs cuBLAS (cublas64_12.dll), which the
    ctranslate2 wheel and the packaged .exe don't ship. That failure surfaces
    mid-session, not at load — which looks like the app 'stopping on its own'.
    So we PROBE with a tiny transcribe here, forcing the CUDA libs to load now;
    if it fails, we rebuild on CPU (which always works)."""
    import numpy as np

    def _build(dev: str, ct: str):
        model = WhisperModel(model_size, device=dev, compute_type=ct)
        # Force the compute libraries to load on a 1s silent clip.
        segments, _info = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
        for _ in segments:
            break
        return model

    try:
        return _build(device, compute_type)
    except Exception as e:
        if (device or "").lower() == "cpu":
            raise
        print(f"[stt] Whisper on device={device!r} can't run ({type(e).__name__}: {e}); "
              f"falling back to CPU. (Set device=cpu in Settings to skip this probe.)")
        return _build("cpu", "int8")


class STTEngine(ABC):
    @abstractmethod
    def stream(self) -> Iterator[Utterance]:
        """Yield Utterance objects as they are finalized, in time order.

        Implementations block between yields as needed (e.g. pacing to real time,
        or waiting on a model). The pipeline consumes this on a background thread.
        """
        raise NotImplementedError
