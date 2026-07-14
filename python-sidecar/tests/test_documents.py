"""Tests for documents.py — file-to-text extraction for Memory uploads."""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documents import extract_text, UnsupportedFileType, FileTooLarge, MAX_FILE_BYTES


def test_extract_txt():
    assert extract_text("notes.txt", b"hello world") == "hello world"


def test_extract_md():
    assert extract_text("notes.md", b"# Title\n\nBody text.") == "# Title\n\nBody text."


def test_extract_txt_decodes_utf8():
    text = "café résumé"
    assert extract_text("notes.txt", text.encode("utf-8")) == text


def test_extract_txt_replaces_invalid_bytes_instead_of_raising():
    # Malformed UTF-8 shouldn't crash the upload — best-effort decode instead.
    result = extract_text("notes.txt", b"valid text \xff\xfe more text")
    assert "valid text" in result


def test_unsupported_extension_raises():
    try:
        extract_text("photo.png", b"whatever")
        assert False, "expected UnsupportedFileType"
    except UnsupportedFileType:
        pass


def test_no_extension_raises_unsupported():
    try:
        extract_text("README", b"whatever")
        assert False, "expected UnsupportedFileType"
    except UnsupportedFileType:
        pass


def test_oversized_file_raises():
    try:
        extract_text("big.txt", b"x" * (MAX_FILE_BYTES + 1))
        assert False, "expected FileTooLarge"
    except FileTooLarge:
        pass


def test_extract_pdf_success():
    fake_page1 = MagicMock(); fake_page1.extract_text.return_value = "Page one text."
    fake_page2 = MagicMock(); fake_page2.extract_text.return_value = "Page two text."
    with patch("pypdf.PdfReader") as MockReader:
        MockReader.return_value.pages = [fake_page1, fake_page2]
        result = extract_text("doc.pdf", b"%PDF-fake-bytes")
    assert "Page one text." in result
    assert "Page two text." in result


def test_extract_pdf_skips_blank_pages():
    fake_page1 = MagicMock(); fake_page1.extract_text.return_value = "Real content."
    fake_page2 = MagicMock(); fake_page2.extract_text.return_value = "   "
    with patch("pypdf.PdfReader") as MockReader:
        MockReader.return_value.pages = [fake_page1, fake_page2]
        result = extract_text("doc.pdf", b"%PDF-fake-bytes")
    assert result == "Real content."


def test_extract_pdf_with_no_text_layer_raises():
    fake_page = MagicMock(); fake_page.extract_text.return_value = ""
    with patch("pypdf.PdfReader") as MockReader:
        MockReader.return_value.pages = [fake_page]
        try:
            extract_text("scanned.pdf", b"%PDF-fake-bytes")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "OCR" in str(e)


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
