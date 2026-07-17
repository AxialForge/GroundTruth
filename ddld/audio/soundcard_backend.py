"""
Live audio capture via `soundcard` (WASAPI on Windows).

One dependency covers both cases you need:
  * LoopbackCapture  — records whatever is *playing* on this PC (a debate on TV,
    a YouTube tab, any app). On Windows this is a WASAPI loopback "microphone"
    exposed for the chosen speaker. This is the case that covers "streaming in a
    browser" and "live TV on the PC" without any per-source wiring.
  * MicrophoneCapture — records a real input device, for an in-person debate.

soundcard hands back float32 frames in [-1, 1] already, which is exactly what
Whisper wants, so there's no dtype conversion on the hot path.

    pip install soundcard
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .base import AudioCapture

_BLOCK_SECONDS = 0.1  # record ~100 ms at a time; the STT layer re-frames to 30 ms for VAD


def _import_soundcard():
    try:
        import soundcard as sc  # noqa: WPS433 (import inside fn: keep it optional)
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "soundcard is not installed. `pip install soundcard` for live capture, "
            "or use stt_engine='faster_whisper' with a .wav file instead."
        ) from e
    return sc


def _record_loop(capture: AudioCapture, device) -> Iterator[np.ndarray]:
    block = max(1, int(capture.sample_rate * _BLOCK_SECONDS))
    with device.recorder(samplerate=capture.sample_rate, channels=1, blocksize=block) as rec:
        while not capture.stopped:
            data = rec.record(numframes=block)          # (n, channels) float32
            if data.ndim > 1:                            # defensive downmix to mono
                data = data.mean(axis=1)
            yield np.ascontiguousarray(data, dtype=np.float32)


class LoopbackCapture(AudioCapture):
    """System audio (what's coming out of the speakers)."""

    def __init__(self, sample_rate: int = 16000, device_id: str = ""):
        super().__init__(sample_rate)
        self._device_id = device_id

    def frames(self) -> Iterator[np.ndarray]:
        sc = _import_soundcard()
        speaker = sc.default_speaker() if not self._device_id else sc.get_speaker(self._device_id)
        if speaker is None:
            raise RuntimeError(f"No speaker found for loopback (device_id={self._device_id!r}).")
        # A speaker's loopback is exposed as an input device sharing its id.
        loop = sc.get_microphone(speaker.id, include_loopback=True)
        yield from _record_loop(self, loop)


class MicrophoneCapture(AudioCapture):
    """A real input device (in-person / headset mic)."""

    def __init__(self, sample_rate: int = 16000, device_id: str = ""):
        super().__init__(sample_rate)
        self._device_id = device_id

    def frames(self) -> Iterator[np.ndarray]:
        sc = _import_soundcard()
        mic = sc.default_microphone() if not self._device_id else sc.get_microphone(self._device_id)
        if mic is None:
            raise RuntimeError(f"No microphone found (device_id={self._device_id!r}).")
        yield from _record_loop(self, mic)


def list_devices() -> dict:
    """Enumerate selectable capture sources for the UI. Never raises — returns
    an 'error' key instead so the front-end can show why the list is empty."""
    try:
        sc = _import_soundcard()
    except RuntimeError as e:
        return {"loopback": [], "mic": [], "error": str(e)}

    def _default_id(dev):
        return getattr(dev, "id", "") if dev else ""

    loopback, mic = [], []
    try:
        default_spk = _default_id(sc.default_speaker())
        for spk in sc.all_speakers():
            loopback.append({"id": spk.id, "name": spk.name, "default": spk.id == default_spk})
    except Exception as e:  # pragma: no cover
        return {"loopback": [], "mic": [], "error": f"enumerating speakers: {e}"}
    try:
        default_mic = _default_id(sc.default_microphone())
        for m in sc.all_microphones(include_loopback=False):
            mic.append({"id": m.id, "name": m.name, "default": m.id == default_mic})
    except Exception as e:  # pragma: no cover
        return {"loopback": loopback, "mic": [], "error": f"enumerating microphones: {e}"}

    return {"loopback": loopback, "mic": mic, "error": ""}
