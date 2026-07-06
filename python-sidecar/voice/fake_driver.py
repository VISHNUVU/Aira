"""Fake voice driver — deterministic, no audio stack. For tests / dev boxes.

Produces a valid silent WAV whose length is proportional to the text, so the
whole voice pipeline (chunking → synth → queue → API) can be exercised in the
Linux sandbox with no TTS engine installed.
"""
from __future__ import annotations

import io
import wave
from typing import Optional

from .base import VoiceDriver, VoiceCapabilities, SynthResult

_SR = 24000


class FakeVoiceDriver(VoiceDriver):
    def __init__(self, wpm: int = 175, **kw):
        self.wpm = wpm

    @property
    def capabilities(self) -> VoiceCapabilities:
        return VoiceCapabilities(
            name="fake",
            offline=True,
            neural=False,
            streaming=True,
            sample_rate=_SR,
            voices=["test"],
            device="cpu",
            notes="Silent WAV generator for tests.",
        )

    def synth(self, text: str, voice: Optional[str] = None,
              rate: float = 1.0, **kw) -> SynthResult:
        words = max(1, len(text.split()))
        seconds = (words / max(1, self.wpm)) * 60.0 / max(0.1, rate)
        nframes = int(seconds * _SR)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SR)
            w.writeframes(b"\x00\x00" * nframes)
        return SynthResult(ok=True, audio=buf.getvalue(), format="wav",
                           sample_rate=_SR, text=text, seconds=seconds)
