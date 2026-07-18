"""
Local STT via faster-whisper. Private, no per-minute cost — fits a homelab.

Phase 0 transcribes a finished .wav file and yields one Utterance per segment.
For a live mic/loopback source (Phase 1), feed rolling ~5s chunks through the
same model with VAD and yield finalized segments — the pipeline downstream is
identical, only the audio front-end changes.

Requires:  pip install faster-whisper
"""
from __future__ import annotations

from typing import Iterator

from ..types import Utterance
from .base import STTEngine


class FasterWhisperSTT(STTEngine):
    def __init__(
        self,
        wav_path: str,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        speaker: "str | None" = None,
    ):
        self.wav_path = wav_path
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.speaker = speaker or None

    def stream(self) -> Iterator[Utterance]:
        from .base import setup_cuda_dll_path
        setup_cuda_dll_path()  # must precede the ctranslate2 import for GPU to load
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is not installed. `pip install faster-whisper` "
                "or use stt_engine='transcript_file' to test without audio."
            ) from e

        from .base import load_whisper_model
        model = load_whisper_model(WhisperModel, self.model_size, self.device, self.compute_type)
        # vad_filter trims silence => fewer empty/garbage utterances.
        segments, _info = model.transcribe(
            self.wav_path,
            vad_filter=True,
            beam_size=5,
        )
        for seg in segments:
            text = seg.text.strip()
            if text:
                yield Utterance(text=text, start=float(seg.start), end=float(seg.end),
                                speaker=self.speaker)
