"""Speech orchestration — turn a streamed reply into continuous natural audio.

This is the glue that lets Aria "talk for a long time, naturally":

    LLM token stream  ─►  SentenceChunker  ─►  synth queue  ─►  audio segments
                                                   │
                                          background worker thread
                                                   │
                                    barge-in / stop() cancels instantly

``SpeechSession`` runs synthesis on a worker thread so the first sentence is
spoken while later sentences are still being generated and rendered — audio
starts in about a second and never stalls between sentences. Each finished
segment is pushed to an output queue that the UI drains (over the HTTP stream)
and plays back-to-back for gapless long-form speech.

Barge-in: ``stop()`` sets a cancel flag the worker checks between segments and
drains the queue, so the assistant goes quiet the instant the user starts
talking or hits stop.

No audio *playback* happens here — the Tauri webview plays the returned audio
(WAV/AIFF) through the browser Audio API. Keeping playback in the UI means this
layer stays pure-Python and testable in the sandbox.
"""
from __future__ import annotations

import base64
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional

from voice import make_voice, SentenceChunker, VoiceDriver
from voice.base import SynthResult


@dataclass
class SpeechSegment:
    """One spoken chunk delivered to the UI."""
    index: int
    text: str
    audio_b64: str
    format: str
    sample_rate: int
    seconds: float

    def to_json(self) -> dict:
        return {
            "index": self.index, "text": self.text, "audio_b64": self.audio_b64,
            "format": self.format, "sample_rate": self.sample_rate,
            "seconds": round(self.seconds, 3),
        }


class SpeechSession:
    """A single speaking turn. Feed it text deltas; drain audio segments."""

    def __init__(self, driver: VoiceDriver, voice: Optional[str] = None,
                 rate: float = 1.0, min_chars: int = 12, max_chars: int = 240):
        self.id = uuid.uuid4().hex[:12]
        self.driver = driver
        self.voice = voice
        self.rate = rate
        self.chunker = SentenceChunker(min_chars=min_chars, max_chars=max_chars)
        self._out: "queue.Queue[Optional[SpeechSegment]]" = queue.Queue()
        self._seg_texts: "queue.Queue[Optional[str]]" = queue.Queue()
        self._cancel = threading.Event()
        self._idx = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # -- producer side (called as the LLM streams) ----------------------
    def feed(self, text: str) -> None:
        if self._cancel.is_set():
            return
        for seg in self.chunker.feed(text):
            self._seg_texts.put(seg)

    def finish(self) -> None:
        """No more text coming; flush the tail and signal end-of-stream."""
        if not self._cancel.is_set():
            for seg in self.chunker.flush():
                self._seg_texts.put(seg)
        self._seg_texts.put(None)     # sentinel

    def stop(self) -> None:
        """Barge-in: cancel synthesis and drain everything."""
        self._cancel.set()
        _drain(self._seg_texts)
        self._seg_texts.put(None)
        _drain(self._out)
        self._out.put(None)

    # -- consumer side (the HTTP handler drains this) -------------------
    def segments(self, timeout: float = 30.0) -> Iterator[SpeechSegment]:
        """Yield audio segments in order until end-of-stream or cancel."""
        while True:
            try:
                seg = self._out.get(timeout=timeout)
            except queue.Empty:
                return
            if seg is None:
                return
            yield seg

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- worker ---------------------------------------------------------
    def _run(self) -> None:
        while not self._cancel.is_set():
            item = self._seg_texts.get()
            if item is None:
                break
            res: SynthResult = self.driver.synth(item, voice=self.voice,
                                                 rate=self.rate)
            if self._cancel.is_set():
                break
            if res.ok and res.audio:
                self._out.put(SpeechSegment(
                    index=self._idx,
                    text=item,
                    audio_b64=base64.b64encode(res.audio).decode("ascii"),
                    format=res.format,
                    sample_rate=res.sample_rate,
                    seconds=res.seconds,
                ))
                self._idx += 1
            # a failed segment is skipped (keeps speech flowing); the text is
            # still shown in the transcript by the UI.
        self._out.put(None)           # end sentinel for consumers


def _drain(q: "queue.Queue") -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


class SpeechManager:
    """Owns the voice driver and tracks the (single) active speaking session.

    Aria speaks one turn at a time; starting a new turn cancels the previous
    one (natural barge-in). Kept deliberately simple and thread-safe.
    """

    def __init__(self, backend: str = "auto", voices_dir: str = "~/.aria/voices",
                 default_voice: Optional[str] = None, rate: float = 1.0):
        self.backend = backend
        self.voices_dir = voices_dir
        self.default_voice = default_voice
        self.rate = rate
        self._driver: Optional[VoiceDriver] = None
        self._active: Optional[SpeechSession] = None
        self._lock = threading.Lock()

    def driver(self) -> VoiceDriver:
        if self._driver is None:
            self._driver = make_voice(self.backend, self.voices_dir)
        return self._driver

    def capabilities(self) -> dict:
        caps = self.driver().capabilities
        return {
            "name": caps.name, "offline": caps.offline, "neural": caps.neural,
            "streaming": caps.streaming, "sample_rate": caps.sample_rate,
            "voices": caps.voices, "device": caps.device, "notes": caps.notes,
            "active_voice": self.default_voice or (caps.voices[0]
                                                   if caps.voices else None),
            "rate": self.rate,
        }

    def start(self, voice: Optional[str] = None,
              rate: Optional[float] = None) -> SpeechSession:
        with self._lock:
            if self._active is not None:
                self._active.stop()
            sess = SpeechSession(self.driver(),
                                 voice=voice or self.default_voice,
                                 rate=rate if rate is not None else self.rate)
            self._active = sess
            return sess

    def stop(self) -> dict:
        with self._lock:
            if self._active is not None:
                self._active.stop()
                sid = self._active.id
                self._active = None
                return {"stopped": True, "session": sid}
            return {"stopped": False}

    def speak_text(self, text: str, voice: Optional[str] = None,
                   rate: Optional[float] = None) -> list[dict]:
        """Convenience: synthesise a whole (non-streamed) string to segments.

        Used by the /speak endpoint and by tests. Blocks until done.
        """
        sess = self.start(voice=voice, rate=rate)
        sess.feed(text)
        sess.finish()
        return [s.to_json() for s in sess.segments()]

    def set_voice(self, voice: Optional[str] = None,
                  rate: Optional[float] = None,
                  backend: Optional[str] = None) -> dict:
        if backend and backend != self.backend:
            self.backend = backend
            self._driver = None
        if voice is not None:
            self.default_voice = voice
        if rate is not None:
            self.rate = rate
        return self.capabilities()
