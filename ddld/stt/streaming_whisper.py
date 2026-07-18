"""
Live streaming STT (Phase 1).

Bridges an AudioCapture (mic / system loopback) to the same Utterance stream the
rest of the pipeline consumes. Nothing downstream changes — this is just a third
STTEngine next to transcript_file and faster_whisper.

How it works
------------
1. Pull float32 blocks from the capture backend.
2. Re-frame into fixed 30 ms frames and run webrtcvad on each.
3. Collect voiced frames; when trailing silence exceeds `silence_ms` (end of an
   utterance) OR the segment hits `max_segment_s` (a long monologue — flush so
   verdicts don't stall), cut the segment.
4. Transcribe that segment with faster-whisper and yield finalized Utterance(s),
   timestamped against a running audio clock.

With an NVIDIA GPU set whisper_device="cuda" (compute_type "float16"): the
'small'/'medium' model transcribes each short segment well inside its duration,
so the feed stays near-real-time.

    pip install faster-whisper webrtcvad-wheels soundcard
"""
from __future__ import annotations

from typing import Iterator, List

import numpy as np

from ..audio.base import AudioCapture
from ..types import Utterance
from .base import STTEngine, load_whisper_model

_FRAME_MS = 30  # webrtcvad accepts only 10 / 20 / 30 ms frames


def _to_pcm16(frame: np.ndarray) -> bytes:
    """float32 [-1,1] -> little-endian int16 bytes for webrtcvad."""
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


class _EnergyGate:
    """Pure-Python speech/non-speech gate — no native dependency.

    Used when webrtcvad isn't available (e.g. the packaged .exe, which ships
    without the native VAD). Tracks an adaptive noise floor: RMS well above the
    floor counts as speech. Coarser than webrtcvad, but false triggers on
    applause/music mostly transcribe to empty/garbage text that's filtered out
    before any API call, so the cost impact is small.
    """

    def __init__(self, sample_rate: int):
        self._floor = 0.004  # initial ambient estimate

    def __call__(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)) + 1e-12))
        # follow quiet fast (so the floor doesn't stay stuck high), rise slowly.
        if rms < self._floor:
            self._floor = 0.9 * self._floor + 0.1 * rms
        else:
            self._floor = 0.995 * self._floor + 0.005 * rms
        return rms > self._floor * 3.5 + 0.004


def _make_speech_gate(aggressiveness: int, sample_rate: int):
    """webrtcvad if importable (better), else the energy gate. Returns a callable
    (frame_float32) -> bool."""
    try:
        import webrtcvad  # optional; absent in the packaged build
    except Exception:
        return _EnergyGate(sample_rate)
    vad = webrtcvad.Vad(aggressiveness)
    return lambda frame: vad.is_speech(_to_pcm16(frame), sample_rate)


class StreamingWhisperSTT(STTEngine):
    def __init__(
        self,
        capture: AudioCapture,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        sample_rate: int = 16000,
        vad_aggressiveness: int = 2,
        silence_ms: int = 700,
        min_segment_ms: int = 400,
        max_segment_s: float = 14.0,
        speaker: "str | None" = None,
    ):
        self.capture = capture
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_ms = silence_ms
        self.min_segment_ms = min_segment_ms
        self.max_segment_s = max_segment_s
        # Optional fixed speaker label (e.g. a single speaker, a named news anchor).
        # Applied to every Utterance this session, so it flows into the feed + exports.
        self.speaker = speaker or None

    # ------------------------------------------------------------------ #
    def stream(self) -> Iterator[Utterance]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Live STT needs faster-whisper. `pip install faster-whisper`."
            ) from e

        model = load_whisper_model(WhisperModel, self.model_size, self.device, self.compute_type)
        is_speech_gate = _make_speech_gate(self.vad_aggressiveness, self.sample_rate)

        frame_len = int(self.sample_rate * _FRAME_MS / 1000)   # 480 samples @ 16 kHz
        frame_s = _FRAME_MS / 1000.0
        max_silence_frames = int(self.silence_ms / _FRAME_MS)
        max_segment_frames = int(self.max_segment_s * 1000 / _FRAME_MS)

        pending = np.empty(0, dtype=np.float32)   # leftover samples between capture blocks
        voiced: List[np.ndarray] = []
        triggered = False
        silence_run = 0
        seg_start = 0.0
        clock = 0.0                                # seconds of audio consumed so far

        # Decouple the sound-card reader from VAD + transcription. Whisper on CPU
        # blocks for a second or more per segment; if that ran on the capture
        # thread the WASAPI recorder would overrun and drop audio ('data
        # discontinuity'). A producer thread keeps draining the card into a queue
        # so no audio is lost while we transcribe; the queue absorbs the burst.
        import queue as _queue
        import threading as _threading

        q: "_queue.Queue" = _queue.Queue()

        def _producer():
            try:
                for blk in self.capture.frames():
                    q.put(blk)
            except Exception as exc:  # surface capture errors to the consumer
                q.put(("__error__", exc))
            finally:
                q.put(None)

        producer = _threading.Thread(target=_producer, name="audio-capture", daemon=True)
        producer.start()

        while True:
            block = q.get()
            if block is None:
                break
            if isinstance(block, tuple) and len(block) == 2 and block[0] == "__error__":
                raise block[1]
            if len(block) == 0:
                continue
            pending = np.concatenate((pending, block)) if pending.size else block

            while pending.size >= frame_len:
                frame = pending[:frame_len]
                pending = pending[frame_len:]
                clock += frame_s

                is_speech = is_speech_gate(frame)

                if not triggered:
                    if is_speech:
                        triggered = True
                        voiced = [frame]
                        silence_run = 0
                        seg_start = clock - frame_s
                    continue

                # inside a segment
                voiced.append(frame)
                silence_run = 0 if is_speech else silence_run + 1

                ended = silence_run >= max_silence_frames
                too_long = len(voiced) >= max_segment_frames
                if ended or too_long:
                    yield from self._finalize(model, voiced, seg_start)
                    triggered = False
                    voiced = []
                    silence_run = 0

        # Source stopped — flush whatever speech is buffered.
        if triggered and voiced:
            yield from self._finalize(model, voiced, seg_start)

    # ------------------------------------------------------------------ #
    def _finalize(self, model, voiced: List[np.ndarray], seg_start: float) -> Iterator[Utterance]:
        duration_ms = len(voiced) * _FRAME_MS
        if duration_ms < self.min_segment_ms:
            return  # too short to be real speech (click, cough)

        audio = np.concatenate(voiced).astype(np.float32)
        # faster-whisper accepts a raw float32 mono array at 16 kHz directly.
        # For continuous listening (news, a long speech), condition_on_previous_text
        # off avoids repetition loops, and the no_speech / logprob thresholds drop
        # Whisper's classic hallucinations on music stings, applause, and silence
        # that slipped past the voice gate — so non-speech never becomes a "claim".
        segments, _info = model.transcribe(
            audio, beam_size=5, vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            if getattr(seg, "no_speech_prob", 0.0) > 0.8 and getattr(seg, "avg_logprob", 0.0) < -0.7:
                continue  # near-certain non-speech; skip the likely hallucination
            start = seg_start + float(seg.start)
            end = seg_start + float(seg.end)
            yield Utterance(text=text, start=start, end=end, speaker=self.speaker)
