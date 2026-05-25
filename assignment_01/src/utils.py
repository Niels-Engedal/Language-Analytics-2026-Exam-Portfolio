"""Small text-cleaning helpers used by the Assignment 1 analysis."""

import unicodedata


def is_symbol_or_punct_only(s: str) -> bool:
    """Return True when a token consists only of Unicode symbols/punctuation."""
    return s != "" and all(unicodedata.category(ch)[0] in {"S", "P"} for ch in s)


def has_weird_unicode(s: str) -> bool:
    """Return True when a token contains control/format characters."""
    return any(unicodedata.category(ch) in {"Cf", "Cc"} for ch in s)
