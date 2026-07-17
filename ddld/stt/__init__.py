"""STT engines + a factory that reads the config."""
from __future__ import annotations

from .base import STTEngine
from .transcript_file import TranscriptFileSTT


def make_stt(cfg, source=None) -> STTEngine:
    """Build the STT engine named in cfg.stt_engine.

    `source` meaning depends on the engine:
      transcript_file -> path to a .txt transcript
      faster_whisper  -> path to a .wav file
      streaming       -> an AudioCapture instance (live mic / loopback); if None,
                         one is built from the config via ddld.audio.make_capture.
    """
    if cfg.stt_engine == "transcript_file":
        return TranscriptFileSTT(source, realtime=cfg.realtime_playback)
    if cfg.stt_engine == "faster_whisper":
        from .faster_whisper_stt import FasterWhisperSTT

        return FasterWhisperSTT(
            source, model_size=cfg.whisper_model_size, device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
        )
    if cfg.stt_engine == "streaming":
        from ..audio import make_capture
        from .streaming_whisper import StreamingWhisperSTT

        capture = source if source is not None else make_capture(cfg)
        return StreamingWhisperSTT(
            capture,
            model_size=cfg.whisper_model_size,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            sample_rate=cfg.sample_rate,
            vad_aggressiveness=cfg.vad_aggressiveness,
            silence_ms=cfg.vad_silence_ms,
            min_segment_ms=cfg.vad_min_segment_ms,
            max_segment_s=cfg.vad_max_segment_s,
        )
    raise ValueError(f"Unknown stt_engine: {cfg.stt_engine!r}")


__all__ = ["STTEngine", "TranscriptFileSTT", "make_stt"]
