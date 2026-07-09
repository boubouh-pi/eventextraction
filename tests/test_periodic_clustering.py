"""Periodic clustering: period detection, rotation, expansion, fallbacks."""
import pandas as pd
import pytest

from app.clustering import cluster_candidates, periodic_clustering


def make_listing(n_events: int, units_per_event: int = 3,
                 nodes_per_unit: int = 1, gap_between_events: int = 4) -> pd.DataFrame:
    """Synthetic periodic listing: each event = [H3 name][Span date][P desc]."""
    tags = ["H3", "Span", "P"] + ["Div"] * max(0, units_per_event - 3)
    rows = []
    order = 100
    parent = 10
    for e in range(n_events):
        for u in range(units_per_event):
            parent += 2
            for _ in range(nodes_per_unit):
                order += 1
                rows.append({
                    "rendering_order": order,
                    "tag": tags[u % len(tags)],
                    "parent_index": parent,
                    "parent_tag": "Div",
                    "depth": 6 + (u % 2),
                    "text_context": f"event {e} unit {u}",
                    "text_length": 15,
                    "letter_ratio": 0.8,
                    "contains_date": int(u == 1),
                    "contains_time": 0,
                    "text_word_location": 0,
                })
        order += gap_between_events
        parent += 3
    df = pd.DataFrame(rows)
    df["row_id"] = df.index
    return df


def test_detects_period_and_separates_all_events():
    nodes = make_listing(n_events=8)
    out = periodic_clustering(nodes, nodes)
    assert out is not None
    # every event in its own cluster, no cluster with two events
    by_cluster = out.groupby("pred_cluster")["text_context"].apply(
        lambda t: {s.split()[1] for s in t}
    )
    assert all(len(events) == 1 for events in by_cluster)
    assert out["pred_cluster"].nunique() == 8


def test_rotation_starts_events_at_heading():
    nodes = make_listing(n_events=6)
    out = periodic_clustering(nodes, nodes)
    firsts = out.sort_values("rendering_order").groupby("pred_cluster").first()
    assert (firsts["tag"] == "H3").all()


def test_returns_none_on_aperiodic_page(node_factory):
    # irregular structure: no repeating unit signature
    rows = [{"rendering_order": i * 7, "tag": t, "parent_index": 100 + i * 3}
            for i, t in enumerate(["H1", "P", "Div", "Span", "Td", "A", "P", "Ul"])]
    nodes = node_factory(rows)
    assert periodic_clustering(nodes, nodes) is None


def test_returns_none_below_min_units(node_factory):
    nodes = node_factory([{"parent_index": i} for i in range(5)])
    assert periodic_clustering(nodes, nodes) is None


def test_segment_expansion_recovers_low_score_nodes():
    nodes = make_listing(n_events=6)
    # withhold one date node from candidates — it must come back via
    # segment-range expansion from the full pool
    candidates = nodes[nodes["text_context"] != "event 2 unit 1"].copy()
    out = periodic_clustering(candidates, nodes)
    assert "event 2 unit 1" in set(out["text_context"])
    recovered = out[out["text_context"] == "event 2 unit 1"].iloc[0]
    same_event = out[out["text_context"] == "event 2 unit 0"].iloc[0]
    assert recovered["pred_cluster"] == same_event["pred_cluster"]


def test_cluster_candidates_prefers_periodic_path():
    nodes = make_listing(n_events=8)
    out = cluster_candidates(nodes, nodes)
    assert out["pred_cluster"].nunique() == 8


def test_cluster_candidates_falls_back_to_gap_chain(node_factory):
    # two far-apart groups, no periodicity → gap-based chain output
    rows = (
        [{"rendering_order": i, "parent_index": 1, "tag": t}
         for i, t in enumerate(["H1", "P", "Div"])]
        + [{"rendering_order": 60 + i, "parent_index": 40, "tag": t}
           for i, t in enumerate(["Ul", "A", "Td"])]
    )
    nodes = node_factory(rows)
    out = cluster_candidates(nodes, nodes)
    assert out["pred_cluster"].nunique() == 2
