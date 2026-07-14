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
        sample_rate = 22050
        # Confirmed live: `--data-format=LEI16@N` only pairs with .caf/.wav/
        # .m4a containers (per `man say`'s own examples) — AIFF has its own
        # implicit format and rejects an explicit override with "Opening
        # output file failed: fmt?", exit code 1. Since every synth() call
        # silently produced zero bytes of audio this way (the caller in
        # speech.py only checks `ok`/`audio`, so a failed segment is just
        # skipped, never surfaced), "Speak replies" was completely silent on
        # this driver — the only backend actually shipping, since Kokoro
        # isn't installed in the build venv. WAV also matches the frontend's
        # default MIME type (audio/wav) and plays in every browser, unlike
        # AIFF which only a WebKit-based view like Tauri's understands.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            cmd = [self._bin(), "-v", voice, "-r", str(wpm),
                   "-o", tmp.name, f"--data-format=LEI16@{sample_rate}", text]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return SynthResult(ok=False, text=text,
                                   error=r.stderr.strip() or "say failed")
            with open(tmp.name, "rb") as f:
                audio = f.read()
            seconds = self._wav_seconds(tmp.name)
            return SynthResult(ok=True, audio=audio, format="wav",
                               sample_rate=sample_rate, text=text,
                               seconds=seconds)
        except VoiceError:
            raise
        except Exception as e:
            return SynthResult(ok=False, text=text, error=str(e))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    @staticmethod
    def _wav_seconds(path: str) -> float:
        import wave
        try:
            with wave.open(path, "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            return 0.0
