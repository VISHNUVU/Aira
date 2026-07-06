"""macOS ``say`` TTS driver — zero-dependency fallback voice.

Every Mac ships the ``say`` command and a set of high-quality system voices
(Siri voices, plus downloadable "Enhanced"/"Premium" voices in System
Settings → Accessibility → Spoken Content). This driver needs nothing installed
and no download, so it's the guaranteed-available fallback if Kokoro isn't set
up yet. It's fully offline.

Not as expressive as Kokoro for very long narration, but the modern macOS
neural system voices (e.g. "Ava", "Zoe", "Jamie") are close, and this driver
makes the "speak" feature work on day one.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from .base import VoiceDriver, VoiceCapabilities, SynthResult, VoiceError


class SayDriver(VoiceDriver):
    def __init__(self, default_voice: str = "Samantha", **kw):
        self.default_voice = default_voice

    def _bin(self) -> str:
        path = shutil.which("say")
        if not path:
            raise VoiceError(
                "The macOS `say` command was not found. This driver only runs "
                "on macOS; use the 'fake' voice on other platforms."
            )
        return path

    @property
    def capabilities(self) -> VoiceCapabilities:
        return VoiceCapabilities(
            name="say",
            offline=True,
            neural=False,      # depends on the chosen system voice
            streaming=False,
            sample_rate=22050,
            voices=self._list_installed_voices(),
            device="cpu",
            notes="Built-in macOS voices. Add Premium/Enhanced voices in "
                  "System Settings → Accessibility → Spoken Content.",
        )

    def _list_installed_voices(self) -> list[str]:
        try:
            out = subprocess.run([self._bin(), "-v", "?"],
                                 capture_output=True, text=True, timeout=5)
            names = []
            for line in out.stdout.splitlines():
                # "Samantha           en_US    # Hello, ..."
                parts = line.split()
                if parts:
                    names.append(parts[0])
            return names or [self.default_voice]
        except Exception:
            return [self.default_voice]

    def synth(self, text: str, voice: Optional[str] = None,
              rate: float = 1.0, **kw) -> SynthResult:
        voice = voice or self.default_voice
        # `say` rate is words-per-minute; map rate multiplier onto ~175 wpm base.
        wpm = max(90, min(360, int(175 * rate)))
        tmp = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
        tmp.close()
        try:
            cmd = [self._bin(), "-v", voice, "-r", str(wpm),
                   "-o", tmp.name, "--data-format=LEI16@22050", text]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return SynthResult(ok=False, text=text,
                                   error=r.stderr.strip() or "say failed")
            with open(tmp.name, "rb") as f:
                audio = f.read()
            return SynthResult(ok=True, audio=audio, format="aiff",
                               sample_rate=22050, text=text)
        except VoiceError:
            raise
        except Exception as e:
            return SynthResult(ok=False, text=text, error=str(e))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
