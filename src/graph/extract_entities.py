"""Extract article-to-article cross references from GDPR text.

GDPR articles frequently cite each other ("in accordance with Article 6",
"Articles 13 and 14", "Articles 44 to 49"). Turning those mentions into
``(:Article)-[:REFERENCES]->(:Article)`` edges is what gives the graph value
beyond plain vector search.

The parser is deliberately conservative: it only follows the explicit list /
range syntax immediately after the word "Article(s)", so paragraph references
like "Article 6(1)" yield article 6 only (the "(1)" is not treated as a
separate article).
"""

from __future__ import annotations

import re

# Matches "Article 6" or "Articles 13, 14 and 15" / "Articles 44 to 49".
# The capture group stops at the first token that is not a number, separator,
# or range word — so "Article 6(1)" captures just "6".
_ARTICLE_REF = re.compile(
    r"Articles?\s+(\d+(?:\s*(?:,|and|or|to)\s*\d+)*)",
    re.IGNORECASE,
)
_RANGE = re.compile(r"(\d+)\s*to\s*(\d+)", re.IGNORECASE)

MAX_ARTICLE = 99  # GDPR has 99 articles; guard against false positives.


def extract_article_references(text: str, source_article: int | None = None) -> list[int]:
    """Return the sorted, de-duplicated article numbers referenced in ``text``.

    ``source_article`` is excluded from the result (an article referencing
    itself is not a useful edge).
    """
    found: set[int] = set()
    if not text:
        return []

    for match in _ARTICLE_REF.finditer(text):
        group = match.group(1)

        # Expand "N to M" ranges inclusively.
        for lo, hi in _RANGE.findall(group):
            lo_i, hi_i = int(lo), int(hi)
            if lo_i <= hi_i:
                found.update(range(lo_i, hi_i + 1))

        # Add every standalone number in the list.
        found.update(int(n) for n in re.findall(r"\d+", group))

    found = {n for n in found if 1 <= n <= MAX_ARTICLE and n != source_article}
    return sorted(found)
