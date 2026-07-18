# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for GroundTruth (one-folder build).

Build:  pyinstaller build/GroundTruth.spec --noconfirm
Output: dist/GroundTruth/GroundTruth.exe

One-folder (not one-file) because faster-whisper/CTranslate2 ship native DLLs that
are far more reliable unpacked than self-extracted on each launch. Wrap dist/ with
Inno Setup (installer/GroundTruth.iss) to get a real Setup.exe.
"""
import os
from PyInstaller.utils.hooks import collect_all

# Paths in a .spec resolve relative to the spec file, so anchor everything at the
# project root (the parent of this build/ dir). SPECPATH is injected by PyInstaller.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

datas = [
    (os.path.join(ROOT, "server", "static"), "server/static"),
    (os.path.join(ROOT, "assets"), "assets"),
]
binaries = []
hiddenimports = []
ICON = os.path.join(ROOT, "assets", "groundtruth.ico")

# Packages with data files / native libs / dynamic imports PyInstaller can miss.
# huggingface_hub + certifi are needed so faster-whisper can DOWNLOAD the model on
# first run (missing them makes Start fail fast, which looks like 'stops on its own').
for pkg in ("faster_whisper", "ctranslate2", "soundcard", "uvicorn", "anthropic",
            "cffi", "huggingface_hub", "certifi", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# CUDA runtime DLLs (from the nvidia-*-cu12 wheels) for GPU transcription. Placed
# at the bundle root (_internal) so setup_cuda_dll_path() — which adds _MEIPASS to
# PATH — lets ctranslate2 load cuBLAS/cuDNN. Adds ~1.3 GB; on a machine without a
# GPU they're simply unused (the app falls back to CPU). No-op if not installed.
import glob as _glob
try:
    import nvidia
    for _root in list(getattr(nvidia, "__path__", [])):
        for _dll in _glob.glob(os.path.join(_root, "*", "bin", "*.dll")):
            binaries.append((_dll, "."))
except Exception:
    pass

a = Analysis(
    [os.path.join(ROOT, "groundtruth_app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # webrtcvad excluded on purpose: its contrib hook is broken for the
    # webrtcvad-wheels distribution, and StreamingWhisperSTT falls back to a
    # pure-Python energy gate when it's absent. tkinter/matplotlib/torch unused.
    excludes=["webrtcvad", "_webrtcvad", "tkinter", "matplotlib", "torch"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="GroundTruth",
    console=True,          # the console doubles as a log; closing it quits the app
    disable_windowed_traceback=False,
    icon=ICON,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="GroundTruth",
)
