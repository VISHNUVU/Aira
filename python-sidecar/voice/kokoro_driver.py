"""Kokoro-82M TTS driver — natural neural voice, fully offline.

Kokoro is an ~82M-param open (Apache-2.0) TTS model. It runs on Apple Silicon
via the ``kokoro`` package (PyTorch/MPS) and produces markedly more natural,
human-sounding prosody than the OS voice — the right default for an assistant
that talks for long stretches. Weights are a ~330 MB download, cached under
``~/.aria/voices`` so it works with no network after first run.

Like the MLX engine driver, ``kokoro`` is imported lazily so the sidecar still
boots on a machine without it (the Linux dev box, a Windows client). Calling
``synth`` without it installed raises a clear ``VoiceError`` telling you what to
install.

Install on the Mac:
    pip install kokoro soundfile
    # espeak-ng is needed for out-of-dictionary words:
    brew install espeak-ng

Implementation status
---------------------
Written against the documented ``kokoro`` API. Never executed here (no audio
stack / GPU in the build sandbox). On the Mac this is verify-and-run: confirm
the ``KPipeline`` import path and that your chosen voice id exists.
"""
from __future__ import annotations

import io
import wave
from typing import Iterator, Optional

from .base import VoiceDriver, VoiceCapabilities, SynthResult, VoiceError

# American + British English voices shipped with Kokoro. 'a' = US, 'b' = UK.
_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]
_SAMPLE_RATE = 24000


class KokoroDriver(VoiceDriver):
    def __init__(self, voices_dir: str, default_voice: str = "af_heart",
                 lang_code: str = "a", **kw):
        self.voices_dir = voices_dir
        self.default_voice = default_voice
        self.lang_code = lang_code          # 'a' US-EN, 'b' UK-EN, ...
        self._pipe = None

    # -- lifecycle ------------------------------------------------------
    def _ensure(self):
        if self._pipe is not None:
            return
        try:
            import os
            os.environ.setdefault("HF_HOME", self.voices_dir)
            from kokoro import KPipeline
        except Exception as e:      # pragma: no cover - needs the package
            raise VoiceError(
                "Kokoro not available. Install with:\n"
                "  pip install kokoro soundfile\n"
                "  brew install espeak-ng\n"
                f"(import error: {e})"
            )
        # repo_id pinned so first run downloads a known-good snapshot.
        self._pipe = KPipeline(lang_code=self.lang_code,
                               repo_id="hexgrad/Kokoro-82M")

    @property
    def capabilities(self) -> VoiceCapabilities:
        return VoiceCapabilities(
            name="kokoro",
            offline=True,
            neural=True,
            streaming=True,
            sample_rate=_SAMPLE_RATE,
            voices=list(_VOICES),
            device="metal",
            notes="Kokoro-82M, ~330MB weights cached in ~/.aria/voices.",
        )

    # -- synthesis ------------------------------------------------------
    def synth(self, text: str, voice: Optional[str] = None,
              rate: float = 1.0, **kw) -> SynthResult:
        self._ensure()
        voice = voice or self.default_voice
        try:
            import numpy as np
            chunks = []
            # KPipeline yields (graphemes, phonemes, audio) per segment.
            for _g, _p, audio in self._pipe(text, voice=voice, speed=rate):
                arr = np.asarray(audio, dtype=np.float32).flatten()
                chunks.append(arr)
            if not chunks:
                return SynthResult(ok=True, audio=b"", text=text,
                                   sample_rate=_SAMPLE_RATE)
            samples = np.concatenate(chunks)
            wav_bytes = _float_to_wav(samples, _SAMPLE_RATE)
            return SynthResult(ok=True, audio=wav_bytes, format="wav",
                               sample_rate=_SAMPLE_RATE, text=text,
                               seconds=len(samples) / _SAMPLE_RATE)
        except VoiceError:
            raise
        except Exception as e:      # pragma: no cover - runtime path
            return SynthResult(ok=False, text=text, error=str(e))

    def stream(self, segments: Iterator[str], voice: Optional[str] = None,
               rate: float = 1.0, **kw) -> Iterator[SynthResult]:
        # Kokoro synthesises fast enough that per-segment synth already gives
        # gapless playback when the service queues results. Keep it simple and
        # correct: one segment -> one SynthResult, in order.
        for seg in segments:
            yield self.synth(seg, voice=voice, rate=rate, **kw)


def _float_to_wav(samples, sample_rate: int) -> bytes:
    """Encode a float32 [-1,1] mono array to 16-bit PCM WAV bytes."""
    import numpy as np
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()
