"""
Raw HTML → DOM-node DataFrame with the training schema.

The training data (``data/raw/*.csv``) was produced by a labeling tool that
emits one row per DOM node with structural, textual and pattern features.
This module reproduces that schema from raw HTML so the *same* LightGBM
model can score live pages.

Design decisions (interview-defensible):
- A "node" is any element carrying **direct text** (its own strings, not
  descendants'). This mirrors the training rows, avoids duplicating the
  same text at every ancestor level, and keeps node counts comparable.
- ``rendering_order`` is the document-order index over *all* elements, so
  gaps between text-bearing nodes reflect intervening markup — the same
  signal the clustering heuristics were tuned on.
- Date/time detection is regex-based (like ``contains_date`` /
  ``contains_time`` in the CSVs): pattern features, not parsing. Parsing
  into canonical dates happens later, at event-assembly time.
"""
from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

# Elements that never carry visible event content.
_SKIP_TAGS = {"script", "style", "noscript", "template", "head", "meta", "link", "title"}

_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_WEEKDAYS = (
    r"mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?"
    r"|fri(?:day)?|sat(?:urday)?|sun(?:day)?"
)

DATE_RE = re.compile(
    rf"""(
        \b\d{{1,2}}[/\-.]\d{{1,2}}[/\-.]\d{{2,4}}\b      # 3/14/2025, 14-03-25
      | \b\d{{4}}-\d{{1,2}}-\d{{1,2}}\b                  # 2025-03-14
      | \b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b # March 14 / mar 14th
      | \b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\b    # 14 March
      | \b(?:{_WEEKDAYS})\b                              # weekday mention
      | \b(?:{_MONTHS})\.?\s+\d{{4}}\b                   # March 2025
    )""",
    re.IGNORECASE | re.VERBOSE,
)

TIME_RE = re.compile(
    r"""(
        \b\d{1,2}:\d{2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b   # 7:00, 19:30, 7:00 pm
      | \b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b                     # 7 pm
      | \b(?:noon|midnight)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Venue-specific words only: "college"/"university" would match event
# names ("Spring College Fair") and mislabel them as locations.
_LOCATION_WORDS = re.compile(
    r"\b(?:location|venue|address|room|hall|center|centre|campus"
    r"|hotel|street|avenue|blvd|building|fairgrounds?)\b",
    re.IGNORECASE,
)


def _direct_text(el: Tag) -> str:
    """Text belonging to this element only (not descendants)."""
    parts = [s for s in el.find_all(string=True, recursive=False)]
    return " ".join(" ".join(parts).split())


def _text_stats(text: str) -> dict:
    n = len(text)
    if n == 0:
        return dict(text_length=0, word_count=0, letter_ratio=0.0,
                    digit_ratio=0.0, whitespace_ratio=0.0)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = sum(c.isspace() for c in text)
    return dict(
        text_length=n,
        word_count=len(text.split()),
        letter_ratio=letters / n,
        digit_ratio=digits / n,
        whitespace_ratio=spaces / n,
    )


def parse_html_to_nodes(html: str) -> pd.DataFrame:
    """
    Parse raw HTML into one row per text-bearing DOM node, with the same
    columns as the labeled training CSVs (minus label/event_id).
    """
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup

    rows: List[dict] = []
    # rendering_order over ALL elements so structural gaps are preserved.
    order = 0
    order_by_el: dict[int, int] = {}

    def walk(el: Tag, depth: int, parent_order: Optional[int], parent_tag: str) -> None:
        nonlocal order
        for child in el.children:
            if not isinstance(child, Tag) or child.name in _SKIP_TAGS:
                continue
            order += 1
            my_order = order
            order_by_el[id(child)] = my_order

            text = _direct_text(child)
            if text:
                attrs = child.attrs or {}
                attr_str = " ".join(
                    f"{k}:{' '.join(v) if isinstance(v, list) else v}"
                    for k, v in attrs.items()
                ).lower()
                stats = _text_stats(text)
                low = text.lower()
                rows.append({
                    "rendering_order": my_order,
                    "tag": child.name.capitalize(),
                    "attributes": attr_str,
                    "text_context": text,
                    "depth": depth,
                    "parent_index": parent_order,
                    "parent_tag": parent_tag,
                    **stats,
                    "contains_date": int(bool(DATE_RE.search(low))),
                    "contains_time": int(bool(TIME_RE.search(low))),
                    "starts_with_digit": int(text[:1].isdigit()),
                    "ends_with_digit": int(text[-1:].isdigit()),
                    "attribute_count": len(attrs),
                    "has_class": int("class" in attrs),
                    "has_id": int("id" in attrs),
                    "attr_has_word_date": int("date" in attr_str),
                    "attr_has_word_time": int("time" in attr_str),
                    "attr_has_word_location": int(
                        "location" in attr_str or "venue" in attr_str or "address" in attr_str
                    ),
                    "text_has_word_date": int("date" in low),
                    "text_word_time": int("time" in low),
                    "text_word_location": int(bool(_LOCATION_WORDS.search(low))),
                })
            walk(child, depth + 1, my_order, child.name.capitalize())

    walk(body, depth=0, parent_order=None, parent_tag="")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["row_id"] = df.index
    return df
