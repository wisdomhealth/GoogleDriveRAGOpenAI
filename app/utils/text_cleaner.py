from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"[ \t]+")
_LINEBREAK_RE = re.compile(r"\n{3,}")
_BROKEN_WORD_RE = re.compile(r"(?<=\w)-\n(?=\w)")


def clean_text(text: str) -> str:
    """Normalize document text while preserving paragraph boundaries."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split by PDF line wrapping while leaving normal hyphenation alone.
    normalized = _BROKEN_WORD_RE.sub("", normalized)
    normalized = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in normalized.split("\n"))
    normalized = _LINEBREAK_RE.sub("\n\n", normalized)
    return normalized.strip()
