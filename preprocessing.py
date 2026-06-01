"""
preprocessing.py
----------------
HTTP parameter normalization pipeline (Section III.B of the paper).

Operations applied in strict order:
1. Iterative URL decoding  (handles double-encoding: %2527 → %27 → ')
2. HTML entity resolution  (&amp;, &#x27;, &#39;, etc.)
3. Unicode full-width mapping (U+FF01–U+FF5E → ASCII)
4. Null byte removal        (\x00)
5. Whitespace collapse      (multiple spaces/tabs/newlines → single space)

Case is deliberately PRESERVED as an obfuscation signal.
"""

import re
import unicodedata
from html import unescape
from urllib.parse import unquote


def url_decode_iterative(text: str, max_iterations: int = 5) -> str:
    """
    Iteratively URL-decode until stable or max_iterations reached.
    Handles double-encoding: %2527 → %27 → '
    """
    prev = None
    for _ in range(max_iterations):
        decoded = unquote(text, encoding='utf-8', errors='replace')
        if decoded == prev:
            break
        prev = text
        text = decoded
    return text


def resolve_html_entities(text: str) -> str:
    """
    Resolve HTML named and numeric entities.
    e.g. &amp; → &,  &#x27; → ',  &#39; → '
    """
    return unescape(text)


def map_fullwidth_to_ascii(text: str) -> str:
    """
    Map Unicode full-width characters (U+FF01–U+FF5E) to ASCII equivalents.
    Neutralizes homoglyph-based evasion (e.g., ＵＮＩＯＮ → UNION).
    """
    result = []
    for ch in text:
        cp = ord(ch)
        # Full-width ASCII variants: U+FF01 (！) to U+FF5E (～)
        if 0xFF01 <= cp <= 0xFF5E:
            result.append(chr(cp - 0xFEE0))
        # Full-width space U+3000
        elif cp == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result)


def remove_null_bytes(text: str) -> str:
    """Remove embedded null bytes used in IDS evasion."""
    return text.replace('\x00', '').replace('%00', '')


def collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters to a single space."""
    return re.sub(r'[\s\t\n\r]+', ' ', text).strip()


def normalize(text: str) -> str:
    """
    Full normalization pipeline.
    Order matters — see paper Section III.B for justification.
    """
    if not isinstance(text, str):
        text = str(text)
    text = url_decode_iterative(text)
    text = resolve_html_entities(text)
    text = map_fullwidth_to_ascii(text)
    text = remove_null_bytes(text)
    text = collapse_whitespace(text)
    return text


if __name__ == '__main__':
    # Smoke tests
    tests = [
        ("%27 OR 1=1 --",          "' OR 1=1 --"),
        ("%2527 OR 1=1",           "' OR 1=1"),          # double-encoded
        ("&amp;amp; SELECT",       "&amp; SELECT"),       # single-pass HTML entity
        ("\uff35\uff2e\uff29\uff2f\uff2e SELECT", "UNION SELECT"),  # full-width
        ("SELECT\x00FROM",         "SELECTFROM"),          # null byte
        ("SELECT    FROM",         "SELECT FROM"),          # whitespace
    ]
    all_pass = True
    for inp, expected in tests:
        got = normalize(inp)
        ok = expected in got or got == expected
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  [{status}] Input: {repr(inp)[:40]:42s} → {repr(got)[:40]}")
    print(f"\n{'All tests passed.' if all_pass else 'SOME TESTS FAILED.'}")
