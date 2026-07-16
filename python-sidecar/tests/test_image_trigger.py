"""Tests for image_trigger.py — trigger phrase detection."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_trigger import extract_image_prompt


def test_matches_common_phrasings():
    assert extract_image_prompt("generate an image of a sunset over mountains") == "a sunset over mountains"
    assert extract_image_prompt("draw a picture of a cat") == "a cat"
    assert extract_image_prompt("create an illustration of a robot") == "a robot"
    assert extract_image_prompt("make me a photo of a beach") == "a beach"
    assert extract_image_prompt("please generate a picture of a forest") == "a forest"
    assert extract_image_prompt("paint an image of the ocean") == "the ocean"


def test_bare_draw_without_image_noun_still_works():
    assert extract_image_prompt("draw a picture of a dragon") == "a dragon"


def test_ignores_ordinary_chat():
    assert extract_image_prompt("how are you today") is None
    assert extract_image_prompt("what's the weather like") is None
    assert extract_image_prompt("") is None
    assert extract_image_prompt(None) is None


def test_ignores_unrelated_generate_phrasing():
    # "generate" alone, without an image/picture/photo noun, shouldn't trigger
    assert extract_image_prompt("generate a report for me") is None


def test_strips_trailing_punctuation():
    assert extract_image_prompt("draw a picture of a horse?") == "a horse"
    assert extract_image_prompt("generate an image of a house.") == "a house"


def test_tolerates_article_glued_onto_noun():
    # Regression: a real live failure where the user typo'd "an aimage"
    # (article glued onto the noun) and the whole deterministic trigger
    # missed, silently falling through to the plain chat model instead of
    # actually generating an image.
    assert extract_image_prompt("Create an aimage of a frog siting on a car") == "a frog siting on a car"
    assert extract_image_prompt("draw a apicture of a dragon") == "a dragon"
    assert extract_image_prompt("generate a aphoto of a beach") == "a beach"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
