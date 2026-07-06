"""Pluggable voice (text-to-speech) layer.

Mirrors the engine layer: everything above this interface is
platform-independent; only concrete drivers know about Kokoro vs the macOS
``say`` command vs a future backend.

The job here is **natural, long-form speech**. Two ideas make that work:

  * ``SentenceChunker`` turns a *stream of text deltas* (the tokens coming off
    the LLM) into a *stream of complete, speakable segments*. TTS starts on the
    first finished sentence — usually within a second — instead of waiting for
    the whole answer, and keeps flowing sentence by sentence. This is what lets
    the assistant "talk for a long time" without a long silence up front and
    without robotic mid-word cut-offs.
  * ``VoiceDriver.synth()`` renders one segment to audio (PCM/WAV bytes). A
    driver may stream or block; the service layer queues segments so playback
    is gapless.

Add a backend by subclassing ``VoiceDriver`` and registering it in
``voice/__init__.py::make_voice``.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class VoiceCapabilities:
    """What a given TTS backend can do on the current platform."""
    name: str
    offline: bool = True          # runs with no network (required for Aria)
    neural: bool = False          # neural prosody vs formant/concatenative
    streaming: bool = False       # can emit audio while still synthesising
    sample_rate: int = 24000
    voices: list[str] = field(default_factory=list)
    device: str = "cpu"           # 'metal' | 'cpu'
    notes: str = ""


@dataclass
class SynthResult:
    """One rendered segment. ``audio`` is raw bytes in ``format`` container."""
    ok: bool
    audio: bytes = b""
    format: str = "wav"           # 'wav' | 'pcm_s16le' | 'aiff'
    sample_rate: int = 24000
    text: str = ""
    seconds: float = 0.0
    error: str = ""


class VoiceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Natural-speech sentence chunker
# ---------------------------------------------------------------------------

# Abbreviations whose trailing period must NOT end a sentence.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg",
    "ie", "no", "vol", "fig", "al", "inc", "ltd", "co", "dept", "univ",
    "approx", "min", "max", "sec", "gen", "gov", "sen", "rep", "phd", "md",
    "a.m", "p.m", "u.s", "u.k", "e.g", "i.e",
}

_SENT_END = re.compile(r"[.!?…]")


def _looks_like_boundary(buf: str, i: int) -> bool:
    """Is the sentence-ender at index ``i`` a real sentence boundary?

    Rejects: decimals (3.14), abbreviations (Dr.), single initials (A.),
    and mid-ellipsis positions. Requires whitespace/end-of-buffer after the
    run of terminal punctuation.
    """
    ch = buf[i]
    # Must be followed by whitespace or end-of-string (allowing closing
    # quotes/brackets right after the punctuation).
    j = i + 1
    while j < len(buf) and buf[j] in "\"')]}»”’":
        j += 1
    if j < len(buf) and not buf[j].isspace():
        # e.g. "3.14" or "www.site" — not a boundary
        return False
    # Look back at the token preceding the period.
    if ch == ".":
        k = i - 1
        word = []
        while k >= 0 and (buf[k].isalnum() or buf[k] == "."):
            word.append(buf[k])
            k -= 1
        token = "".join(reversed(word)).strip(".").lower()
        if token in _ABBREV:
            return False
        # single-letter initial: "J." in "J. Smith"
        if len(token) == 1 and token.isalpha():
            return False
        # decimal: digit before AND after the dot ("3.14")
        if (i - 1 >= 0 and buf[i - 1].isdigit()
                and i + 1 < len(buf) and buf[i + 1].isdigit()):
            return False
    return True


class SentenceChunker:
    """Accumulate streamed text and emit complete, speakable segments.

    Usage::

        ch = SentenceChunker()
        for delta in llm_token_stream():
            for seg in ch.feed(delta):
                tts.speak(seg)
        for seg in ch.flush():
            tts.speak(seg)

    Design goals for natural long-form speech:
      * emit as soon as a real sentence closes (low latency to first audio);
      * never split inside abbreviations, decimals, or initials;
      * also break on paragraph/newline boundaries;
      * merge tiny fragments up to ``min_chars`` so we don't speak "Yes."
        as its own choppy clip;
      * hard-wrap a runaway sentence at the last comma/clause before
        ``max_chars`` so a very long sentence still streams in pieces.
    """

    def __init__(self, min_chars: int = 12, max_chars: int = 240):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, text: str) -> Iterator[str]:
        self._buf += text
        yield from self._drain(final=False)

    def flush(self) -> Iterator[str]:
        yield from self._drain(final=True)
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            yield tail

    # -- internal -------------------------------------------------------
    def _drain(self, final: bool) -> Iterator[str]:
        while True:
            seg = self._next_segment(final)
            if seg is None:
                return
            seg = seg.strip()
            if seg:
                yield seg

    def _next_segment(self, final: bool) -> Optional[str]:
        buf = self._buf
        if not buf:
            return None

        # 1) hard paragraph break
        nl = buf.find("\n\n")
        if nl != -1:
            seg, self._buf = buf[:nl], buf[nl + 2:]
            return seg

        # 2) sentence boundary at/after min_chars
        for m in _SENT_END.finditer(buf):
            i = m.start()
            # consume a run of terminal punctuation ("?!", "...")
            end = i
            while end + 1 < len(buf) and buf[end + 1] in ".!?…":
                end += 1
            if end + 1 < len(buf) or final:
                if _looks_like_boundary(buf, end) and (end + 1) >= self.min_chars:
                    seg, self._buf = buf[:end + 1], buf[end + 1:]
                    return seg

        # 3) runaway sentence: wrap at last comma/clause before max_chars
        if len(buf) >= self.max_chars:
            window = buf[:self.max_chars]
            cut = max(window.rfind(", "), window.rfind("; "),
                      window.rfind(" — "), window.rfind(": "))
            if cut <= 0:
                cut = window.rfind(" ")      # last resort: word boundary
            if cut > 0:
                seg, self._buf = buf[:cut + 1], buf[cut + 1:]
                return seg
        return None


# ---------------------------------------------------------------------------
# Driver interface
# ---------------------------------------------------------------------------

class VoiceDriver(ABC):
    """One concrete TTS backend."""

    @property
    @abstractmethod
    def capabilities(self) -> VoiceCapabilities: ...

    @abstractmethod
    def synth(self, text: str, voice: Optional[str] = None,
              rate: float = 1.0, **kw) -> SynthResult:
        """Render ``text`` to audio bytes. Blocking."""

    def stream(self, segments: Iterator[str], voice: Optional[str] = None,
               rate: float = 1.0, **kw) -> Iterator[SynthResult]:
        """Render a stream of segments in order. Default: synth each in turn.

        Drivers with true streaming synthesis may override for lower latency.
        """
        for seg in segments:
            yield self.synth(seg, voice=voice, rate=rate, **kw)

    def list_voices(self) -> list[str]:
        return list(self.capabilities.voices)
