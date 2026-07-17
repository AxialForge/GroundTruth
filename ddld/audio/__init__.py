"""Live audio capture backends + a factory that reads the config."""
from __future__ import annotations

from .base import AudioCapture


def make_capture(cfg, source: str = "", device_id: str = "") -> AudioCapture:
    """Build the capture backend for `source` ('loopback' | 'mic').

    Falls back to cfg.audio_source / cfg.audio_device_id when not given, so the
    headless path and the server can both call this the same way.
    """
    source = (source or cfg.audio_source or "loopback").lower()
    device_id = device_id or cfg.audio_device_id
    from .soundcard_backend import LoopbackCapture, MicrophoneCapture

    if source == "loopback":
        return LoopbackCapture(sample_rate=cfg.sample_rate, device_id=device_id)
    if source in ("mic", "microphone"):
        return MicrophoneCapture(sample_rate=cfg.sample_rate, device_id=device_id)
    raise ValueError(f"Unknown audio source: {source!r} (expected 'loopback' or 'mic').")


def list_devices() -> dict:
    from .soundcard_backend import list_devices as _list

    return _list()


__all__ = ["AudioCapture", "make_capture", "list_devices"]
