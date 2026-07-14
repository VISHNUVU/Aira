"""File ingestion for Memory — turns an uploaded file into plain text ready
for memory.ingest_text(). Pure text extraction only: no OCR, no layout
analysis, nothing that would pull in a heavy dependency for what's
fundamentally "get the words out of this file."

Supported: .txt, .md (decoded directly) and .pdf (pypdf's text layer).
Anything else raises UnsupportedFileType — the caller turns that into a
clean 400, not a stack trace.
"""
from __future__ import annotations

import io
import os

MAX_FILE_BYTES = 25 * 1024 * 1024  # 25MB — comfortably past any real note/doc, guards against pasting a stray video as base64 by mistake


class UnsupportedFileType(ValueError):
    pass


class FileTooLarge(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Returns the extracted plain text, or raises UnsupportedFileType /
    FileTooLarge. Dispatches purely on file extension — content-sniffing
    would be more robust but also more surface area for what's meant to be
    a small, predictable feature."""
    if len(data) > MAX_FILE_BYTES:
        raise FileTooLarge(
            f"{filename} is {len(data) / 1_048_576:.1f}MB — the limit is "
            f"{MAX_FILE_BYTES / 1_048_576:.0f}MB")

    ext = os.path.splitext(filename)[1].lower()
    if ext in (".txt", ".md", ".markdown"):
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        return _extract_pdf(data)
    raise UnsupportedFileType(
        f"unsupported file type '{ext or filename}' — only .txt, .md, and .pdf are supported")


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise UnsupportedFileType(
            "PDF support isn't available in this build (pypdf missing)")
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise ValueError(
            "couldn't extract any text from this PDF — it may be scanned "
            "images with no text layer (OCR isn't supported)")
    return text
