"""HTML → node-frame parser: schema fidelity and pattern features."""
import pandas as pd

from app.html_parser import DATE_RE, TIME_RE, parse_html_to_nodes


def test_parses_nodes_with_training_schema(sample_html):
    df = parse_html_to_nodes(sample_html)

    assert not df.empty
    required = {
        "rendering_order", "tag", "parent_tag", "depth", "parent_index",
        "text_context", "text_length", "word_count", "letter_ratio",
        "digit_ratio", "whitespace_ratio", "contains_date", "contains_time",
        "starts_with_digit", "ends_with_digit", "has_class", "has_id",
        "attr_has_word_date", "attr_has_word_time", "attr_has_word_location",
        "text_has_word_date", "text_word_time", "text_word_location", "row_id",
    }
    assert required.issubset(df.columns)


def test_date_and_time_pattern_features(sample_html):
    df = parse_html_to_nodes(sample_html)

    date_nodes = df[df["contains_date"] == 1]
    time_nodes = df[df["contains_time"] == 1]
    assert "March 14, 2025" in set(date_nodes["text_context"])
    assert "6:00 pm - 8:30 pm" in set(time_nodes["text_context"])


def test_attr_hints_detected(sample_html):
    df = parse_html_to_nodes(sample_html)
    by_text = df.set_index("text_context")
    assert by_text.loc["March 14, 2025", "attr_has_word_date"] == 1
    assert by_text.loc["Sacramento Convention Center", "attr_has_word_location"] == 1


def test_rendering_order_monotonic_and_parent_links(sample_html):
    df = parse_html_to_nodes(sample_html)
    assert df["rendering_order"].is_monotonic_increasing
    # every non-root node's parent_index refers to an earlier element
    with_parent = df[df["parent_index"].notna()]
    assert (with_parent["parent_index"] < with_parent["rendering_order"]).all()


def test_script_and_style_ignored():
    html = "<body><script>var d='March 14, 2025';</script><p>hello world</p></body>"
    df = parse_html_to_nodes(html)
    assert list(df["text_context"]) == ["hello world"]


def test_empty_html_returns_empty_frame():
    assert parse_html_to_nodes("<html></html>").empty


def test_date_regex_variants():
    positives = ["3/14/2025", "2025-03-14", "March 14th", "14 March",
                 "mar 14", "Saturday", "thu", "October 2025"]
    for s in positives:
        assert DATE_RE.search(s.lower()), s
    assert not DATE_RE.search("hello world")


def test_time_regex_variants():
    for s in ["7:00", "19:30:00", "7 pm", "7:00 a.m.", "noon"]:
        assert TIME_RE.search(s.lower()), s
    assert not TIME_RE.search("we meet often")
