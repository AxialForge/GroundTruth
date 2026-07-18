"""
STT engine interface. Everything the pipeline needs from speech-to-text is
`stream()` yielding finalized Utterances. Swap cloud/local behind this ABC.
"""
from __future__ import annotations

import os
import shutil
import sys
from abc import ABC, abstractmethod
from typing import Iterator

from ..types import Utterance


_CUDA_PATH_DONE = False


def setup_cuda_dll_path() -> None:
    """Put the CUDA runtime DLLs on PATH so ctranslate2 can use the GPU.

    ctranslate2 loads cublas64_12.dll via a plain LoadLibrary that only honors
    PATH (not os.add_dll_directory), and cuBLAS pulls in cudart/cudnn — so all the
    CUDA DLLs must be reachable via PATH. In the packaged app they're bundled into
    _internal; from source they come from the nvidia-*-cu12 pip wheels. Must run
    BEFORE faster_whisper/ctranslate2 is imported. No GPU / no libs → harmless;
    load_whisper_model still falls back to CPU."""
    global _CUDA_PATH_DONE
    if _CUDA_PATH_DONE:
        return
    _CUDA_PATH_DONE = True

    dirs = []
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        dirs.append(base)                       # DLLs bundled alongside ctranslate2.dll
        dirs.append(os.path.join(base, "cuda"))
    else:
        try:
            import nvidia  # namespace package from the nvidia-*-cu12 wheels
            for root in list(getattr(nvidia, "__path__", [])):
                for sub in os.listdir(root):
                    b = os.path.join(root, sub, "bin")
                    if os.path.isdir(b):
                        dirs.append(b)
        except Exception:
            pass

    for d in dirs:
        if d and os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


# Per-process memo of what device actually worked, so repeated Starts don't
# re-probe a GPU we already know can't run inference (and don't load the model
# twice each time).
_RESOLVED_DEVICE: dict = {}


def _app_model_root() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    root = (os.path.join(base, "GroundTruth", "models") if os.environ.get("APPDATA")
            else os.path.join(base, ".groundtruth", "models"))
    os.makedirs(root, exist_ok=True)
    return root


def materialize_model(model_size: str) -> str:
    """Return a directory holding the faster-whisper model as REAL FILES.

    The HuggingFace cache stores model.bin as a Windows symlink into blobs/, and
    the packaged app's CTranslate2 can't open that symlink ('Unable to open file
    model.bin') even though the same call works from a normal Python run. We
    resolve the symlinks ONCE into GroundTruth's own data dir (reusing the already
    downloaded blob — no re-download), and load plain files from there.

    If `model_size` is already a path to a real model dir, it's returned as-is."""
    if os.path.isdir(model_size) and os.path.isfile(os.path.join(model_size, "model.bin")):
        return model_size

    local = os.path.join(_app_model_root(), f"faster-whisper-{model_size}")
    binp = os.path.join(local, "model.bin")
    if os.path.isfile(binp) and not os.path.islink(binp) and os.path.getsize(binp) > 1_000_000:
        return local  # already materialized as real files

    os.makedirs(local, exist_ok=True)
    setup_cuda_dll_path()
    from faster_whisper import download_model
    cache_dir = download_model(model_size)  # HF snapshot (downloads if missing); symlinked on Windows
    for name in os.listdir(cache_dir):
        src = os.path.join(cache_dir, name)
        # Resolve the blob WITHOUT traversing the symlink: os.path.realpath (and
        # CTranslate2's open, and os.path.isdir/isfile which stat-follow) raise
        # WinError 448 'untrusted mount point' on the HF symlinks in the packaged
        # app. os.readlink reads the reparse target string without traversing.
        if os.path.islink(src):
            target = os.readlink(src)                       # e.g. '..\\..\\blobs\\<sha>'
            blob = target if os.path.isabs(target) else os.path.normpath(
                os.path.join(os.path.dirname(src), target))
        elif os.path.isfile(src):
            blob = src                                       # already a real file
        else:
            continue                                         # a real subdirectory
        dst = os.path.join(local, name)
        tmp = dst + ".part"
        shutil.copyfile(blob, tmp)                           # blob is a plain file — safe to open
        os.replace(tmp, dst)                                 # atomic swap into place
    return local


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
    import numpy as np

    # Real, symlink-free model path — the fix for the packaged app's
    # 'Unable to open file model.bin'.
    model_path = materialize_model(model_size)

    def _build(dev: str, ct: str):
        model = WhisperModel(model_path, device=dev, compute_type=ct)
        segments, _info = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
        for _ in segments:
            break
        return model

    key = (model_path, (device or "").lower(), (compute_type or "").lower())
    resolved = _RESOLVED_DEVICE.get(key)
    if resolved:
        return _build(*resolved)

    try:
        model = _build(device, compute_type)
        _RESOLVED_DEVICE[key] = (device, compute_type)
        return model
    except Exception as e:
        if (device or "").lower() == "cpu":
            raise
        print(f"[stt] Whisper on device={device!r} can't run ({type(e).__name__}: {e}); "
              f"falling back to CPU. (Set device=cpu in Settings to skip this probe.)")
        model = _build("cpu", "int8")
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
