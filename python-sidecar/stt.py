"""Speech-to-text — offline dictation for the chat composer's mic button.

Uses `mlx_audio.stt` (Whisper on Metal). Import is lazy so the rest of the
app still runs on machines without mlx-audio (e.g. non-Apple-Silicon).

Model choice: the "-asr-" mlx-community conversions ship a full HuggingFace
processor (tokenizer + preprocessor_config.json); the plain "-mlx" whisper
conversions don't, and mlx_audio's tokenizer loader raises `ValueError:
Processor not found` on those. Default to the tiny 4-bit ASR variant — fast
enough for live dictation, small enough not to make the mic button itself a
multi-GB download.
"""
from __future__ import annotations

DEFAULT_MODEL = "mlx-community/whisper-tiny-asr-4bit"


class SttEngine:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._model = None

    @property
    def available(self) -> bool:
        try:
            import mlx_audio.stt  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from mlx_audio.stt.utils import load_model
            except ImportError:
                raise RuntimeError(
                    "mlx-audio isn't available in this build — dictation "
                    "needs it even though chat inference doesn't. Check "
                    "`available` before calling transcribe()."
                )
            self._model = load_model(self.model_id)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a local audio file (WAV; PCM formats miniaudio reads
        directly — no ffmpeg dependency). Returns the trimmed text."""
        self._ensure_loaded()
        from mlx_audio.stt.generate import generate_transcription
        result = generate_transcription(model=self._model, audio=audio_path, verbose=False)
        text = result.text if hasattr(result, "text") else str(result)
        return text.strip()
