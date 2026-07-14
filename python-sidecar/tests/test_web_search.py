"""Tests for the web_search module — trigger detection + result parsing."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_search import extract_search_query, search_web

# A minimal fragment matching DuckDuckGo's html.duckduckgo.com result markup.
FAKE_DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/one">Example One</a>
  <a class="result__snippet">First result snippet text.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/two">Example &amp; Two</a>
  <a class="result__snippet">Second &lt;result&gt; snippet.</a>
</div>
"""


def test_extract_search_query_matches_common_phrasings():
    assert extract_search_query("search for the weather in tokyo") == "the weather in tokyo"
    assert extract_search_query("search the web for latest macos release") == "latest macos release"
    assert extract_search_query("look up python 3.13 changelog") == "python 3.13 changelog"
    assert extract_search_query("google the capital of mongolia") == "the capital of mongolia"
    assert extract_search_query("please search: rust async book") == "rust async book"


def test_extract_search_query_ignores_ordinary_chat():
    assert extract_search_query("how are you today") is None
    assert extract_search_query("what's the weather like") is None
    assert extract_search_query("") is None
    assert extract_search_query(None) is None


def test_extract_search_query_strips_trailing_punctuation():
    assert extract_search_query("search for best pizza near me?") == "best pizza near me"


def test_search_web_empty_query_short_circuits():
    assert search_web("") == []
    assert search_web("   ") == []


def test_search_web_parses_results():
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return FAKE_DDG_HTML.encode("utf-8")

    with patch("web_search.urllib.request.urlopen", return_value=FakeResp()):
        results = search_web("test query", k=5)
    assert len(results) == 2
    assert results[0] == {"title": "Example One", "url": "https://example.com/one",
                          "snippet": "First result snippet text."}
    assert results[1]["title"] == "Example & Two"
    assert results[1]["snippet"] == "Second <result> snippet."


def test_search_web_respects_k_limit():
    with patch("web_search.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = FAKE_DDG_HTML.encode()
        results = search_web("test query", k=1)
    assert len(results) == 1


def test_search_web_network_failure_returns_empty():
    with patch("web_search.urllib.request.urlopen", side_effect=OSError("no network")):
        assert search_web("anything") == []


def test_search_web_malformed_html_returns_empty():
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"<html><body>no results here</body></html>"

    with patch("web_search.urllib.request.urlopen", return_value=FakeResp()):
        assert search_web("test query") == []


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
