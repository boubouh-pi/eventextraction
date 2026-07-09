import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


SAMPLE_HTML = """
<html><head><title>Fairs</title></head><body>
<nav><a href="/">Home</a><a href="/about">About us</a></nav>
<main>
  <h1>Spring College Fairs</h1>
  <div class="event">
    <h2>Sacramento College Fair</h2>
    <span class="date">March 14, 2025</span>
    <span class="time">6:00 pm - 8:30 pm</span>
    <span class="location">Sacramento Convention Center</span>
  </div>
  <div class="event">
    <h2>Fresno College Fair</h2>
    <span class="date">April 2, 2025</span>
    <span class="time">5:30 pm - 8:00 pm</span>
    <span class="location">Fresno Fairgrounds Hall</span>
  </div>
</main>
<footer><p>Copyright 2025 - contact us</p></footer>
</body></html>
"""


@pytest.fixture
def sample_html() -> str:
    return SAMPLE_HTML


def make_nodes(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal node frame for clustering tests."""
    defaults = {
        "tag": "Span", "parent_tag": "Div", "depth": 5, "parent_index": 1,
        "text_context": "x", "text_length": 10, "letter_ratio": 0.8,
        "contains_date": 0, "contains_time": 0, "text_word_location": 0,
    }
    out = []
    for i, r in enumerate(rows):
        row = {**defaults, **r}
        row.setdefault("rendering_order", i * 2)
        row.setdefault("row_id", i)
        out.append(row)
    return pd.DataFrame(out)


@pytest.fixture
def node_factory():
    return make_nodes
