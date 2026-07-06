"""Voice layer tests — sentence chunking, fake TTS, speech session, API.

Runs fully in the sandbox (no audio stack): the fake voice driver emits silent
WAV so the whole pipeline is exercised deterministically.
"""
import os
import sys
import wave
import io
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voice import make_voice, auto_voice_name, SentenceChunker
from voice.base import _looks_like_boundary
from speech import SpeechManager, SpeechSession

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def _feed_all(text, min_chars=12, max_chars=240):
    """Feed text one character at a time (worst case for a streaming chunker)."""
    ch = SentenceChunker(min_chars=min_chars, max_chars=max_chars)
    out = []
    for c in text:
        out.extend(ch.feed(c))
    out.extend(ch.flush())
    return out


def test_chunker_basic_sentences():
    segs = _feed_all("Hello there. How are you? I am well!")
    check("splits three sentences", len(segs) == 3)
    check("first sentence intact", segs[0] == "Hello there.")
    check("question kept whole", segs[1] == "How are you?")
    # nothing lost: rejoin equals original (modulo the spaces we split on)
    joined = " ".join(segs)
    check("no text lost", joined == "Hello there. How are you? I am well!")


def test_chunker_abbreviations():
    segs = _feed_all("Dr. Smith met Mr. Jones at 3 p.m. today. They talked.")
    check("does not split on Dr./Mr./p.m.", len(segs) == 2)
    check("abbrev sentence whole",
          segs[0] == "Dr. Smith met Mr. Jones at 3 p.m. today.")


def test_chunker_decimals():
    segs = _feed_all("The value is 3.14 exactly. Pi is irrational.")
    check("does not split on decimal", len(segs) == 2)
    check("decimal preserved", "3.14" in segs[0])


def test_chunker_initials():
    segs = _feed_all("Written by J. R. Tolkien himself. A classic.")
    check("does not split on initials", len(segs) == 2)


def test_chunker_streaming_equiv():
    text = "First point here. Second point follows! And a third?"
    per_char = _feed_all(text)
    ch = SentenceChunker()
    whole = list(ch.feed(text)) + list(ch.flush())
    check("char-stream == whole-string chunking", per_char == whole)


def test_chunker_runaway_sentence():
    long = ("This is a very long sentence that just keeps going and going "
            "with many clauses, and more clauses, and still more detail, "
            "and yet even more elaboration until it finally stops here.")
    segs = _feed_all(long, max_chars=80)
    check("runaway sentence is split", len(segs) >= 2)
    check("runaway pieces rejoin",
          "".join(segs).replace(" ", "") == long.replace(" ", ""))


def test_chunker_boundary_helper():
    check("period+space is boundary", _looks_like_boundary("Hi there. Ok", 8))
    check("Dr. is not boundary", not _looks_like_boundary("Dr. Smith", 2))


def test_fake_driver_wav():
    v = make_voice("fake")
    r = v.synth("Hello world this is a test.")
    check("fake synth ok", r.ok and len(r.audio) > 0)
    # valid WAV?
    w = wave.open(io.BytesIO(r.audio), "rb")
    check("valid mono 16-bit wav",
          w.getnchannels() == 1 and w.getsampwidth() == 2)
    check("duration scales with words", r.seconds > 0)


def test_auto_voice_name():
    name = auto_voice_name()
    check("auto voice picks known backend", name in ("kokoro", "say", "fake"))


def test_speech_manager_speak_text():
    mgr = SpeechManager(backend="fake")
    segs = mgr.speak_text("One sentence here. Two sentences here. Three now.")
    check("speak_text returns 3 segments", len(segs) == 3)
    check("segments ordered", [s["index"] for s in segs] == [0, 1, 2])
    check("each segment carries audio",
          all(len(base64.b64decode(s["audio_b64"])) > 0 for s in segs))
    caps = mgr.capabilities()
    check("capabilities report offline", caps["offline"] is True)


def test_speech_session_barge_in():
    mgr = SpeechManager(backend="fake")
    sess = mgr.start()
    sess.feed("A long first sentence to speak. ")
    sess.stop()                      # barge-in before finishing
    check("session cancelled after stop", sess.cancelled)
    # starting a new turn cancels the old one
    s2 = mgr.start()
    check("new session id differs", s2.id != sess.id)
    stopped = mgr.stop()
    check("manager stop reports session", stopped["stopped"] is True)


def test_route_integration():
    # Hit the voice routes through the same _route the HTTP server uses.
    from app import SidecarService, _route
    import tempfile
    home = tempfile.mkdtemp(prefix="aria_voice_")
    os.environ["ARIA_VOICE"] = "fake"
    svc = SidecarService(engine_name="fake", home=home, use_lance=False)

    st, body = _route(svc, "GET", "/voice", {})
    check("/voice returns 200", st == 200 and "voices" in body)

    st, body = _route(svc, "POST", "/speak",
                      {"text": "Hello there friend. Goodbye for now."})
    check("/speak returns segments", st == 200 and body["count"] == 2)

    st, body = _route(svc, "POST", "/voice/set", {"rate": 1.2})
    check("/voice/set updates rate", st == 200 and body["rate"] == 1.2)

    st, body = _route(svc, "POST", "/chat/speak",
                      {"messages": [{"role": "user", "content": "hello"}]})
    check("/chat/speak returns text+speech",
          st == 200 and "content" in body and "speech" in body)

    st, body = _route(svc, "POST", "/voice/stop", {})
    check("/voice/stop returns 200", st == 200)


if __name__ == "__main__":
    for fn in [
        test_chunker_basic_sentences, test_chunker_abbreviations,
        test_chunker_decimals, test_chunker_initials,
        test_chunker_streaming_equiv, test_chunker_runaway_sentence,
        test_chunker_boundary_helper, test_fake_driver_wav,
        test_auto_voice_name, test_speech_manager_speak_text,
        test_speech_session_barge_in, test_route_integration,
    ]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{'='*40}\nvoice: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
