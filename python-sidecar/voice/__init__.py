"""Voice (TTS) layer: pluggable, natural-speech backends.

Use ``make_voice(name, ...)`` to construct the right driver for the platform.
``SentenceChunker`` (in ``voice.base``) turns a streamed reply into speakable
segments so the assistant starts talking within ~1s and flows naturally for
arbitrarily long answers.
"""
from __future__ import annotations

import platform

from .base import (
    VoiceDriver,
    VoiceCapabilities,
    SynthResult,
    VoiceError,
    SentenceChunker,
)

__all__ = [
    "VoiceDriver", "VoiceCapabilities", "SynthResult", "VoiceError",
    "SentenceChunker", "make_voice", "auto_voice_name",
]


def auto_voice_name() -> str:
    """Best natural voice available on this machine, preferring neural+offline.

    Order: Kokoro (neural, cross-platform) if importable, else macOS ``say`` on
    a Mac, else the fake driver. The service falls back gracefully at runtime,
    so this is only the *preferred* pick.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("kokoro") is not None:
            return "kokoro"
    except Exception:
        pass
    if platform.system() == "Darwin":
        return "say"
    return "fake"


def make_voice(name: str, voices_dir: str = "", **kw) -> VoiceDriver:
    """Factory. ``name`` in {'kokoro','say','fake','auto'}."""
    if name == "auto":
        name = auto_voice_name()
    if name == "kokoro":
        from .kokoro_driver import KokoroDriver
        return KokoroDriver(voices_dir or "~/.aria/voices", **kw)
    if name == "say":
        from .say_driver import SayDriver
        return SayDriver(**kw)
    if name == "fake":
        from .fake_driver import FakeVoiceDriver
        return FakeVoiceDriver(**kw)
    raise VoiceError(f"unknown voice backend: {name!r}")
