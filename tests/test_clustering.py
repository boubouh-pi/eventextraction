"""Clustering heuristics: separation, mega-cluster splitting, merging, noise."""
import pandas as pd

from app.clustering import (
    cluster_candidates,
    merge_tiny_subclusters,
    remove_obvious_noise,
    sub_cluster_by_event_boundaries,
    unified_clustering,
)


def test_unified_clustering_splits_on_gap(node_factory):
    # Two groups separated by a large rendering + parent gap.
    rows = (
        [{"rendering_order": i, "parent_index": 1, "depth": 5} for i in range(3)]
        + [{"rendering_order": 50 + i, "parent_index": 40, "depth": 5} for i in range(3)]
    )
    nodes = node_factory(rows)
    clustered = unified_clustering(nodes)
    assert clustered["pred_cluster"].nunique() == 2


def test_unified_clustering_keeps_adjacent_nodes_together(node_factory):
    rows = [{"rendering_order": i, "parent_index": 1, "depth": 5} for i in range(6)]
    clustered = unified_clustering(node_factory(rows))
    assert clustered["pred_cluster"].nunique() == 1


def test_sub_cluster_splits_mega_cluster_at_date_boundaries(node_factory):
    # A 12-node mega-cluster: 4 repeating (date, name, location) triplets.
    rows = []
    for k in range(4):
        rows.append({"rendering_order": k * 3, "contains_date": 1})
        rows.append({"rendering_order": k * 3 + 1})
        rows.append({"rendering_order": k * 3 + 2})
    nodes = node_factory(rows)
    nodes["pred_cluster"] = 1

    result = sub_cluster_by_event_boundaries(nodes)
    assert result["pred_cluster"].nunique() == 4


def test_sub_cluster_leaves_small_clusters_untouched(node_factory):
    nodes = node_factory([{"contains_date": 1} for _ in range(4)])
    nodes["pred_cluster"] = 1
    result = sub_cluster_by_event_boundaries(nodes)
    assert result["pred_cluster"].nunique() == 1


def test_merge_tiny_subclusters(node_factory):
    rows = (
        [{"rendering_order": i, "parent_index": 1} for i in range(5)]
        + [{"rendering_order": 6, "parent_index": 1}]  # orphan fragment
    )
    nodes = node_factory(rows)
    nodes["pred_cluster"] = [1] * 5 + [2]
    result = merge_tiny_subclusters(nodes, min_size=3)
    assert result["pred_cluster"].nunique() == 1


def test_merge_refuses_combining_two_events(node_factory):
    # Tiny cluster with 2 dates next to a cluster with 2 dates: merging
    # would fuse two distinct events -> must be refused.
    rows = [{"rendering_order": i, "contains_date": 1} for i in range(4)]
    nodes = node_factory(rows)
    nodes["pred_cluster"] = [1, 1, 2, 2]
    result = merge_tiny_subclusters(nodes, min_size=3)
    assert result["pred_cluster"].nunique() == 2


def test_remove_obvious_noise(node_factory):
    nodes = node_factory([
        {"text_context": "Sacramento College Fair", "text_length": 23, "depth": 6},
        {"text_context": "Contact us today", "text_length": 16, "depth": 6},
        {"text_context": "abc", "text_length": 3, "depth": 6},   # fragment
        {"text_context": "Site header", "text_length": 11, "depth": 1},  # shallow
    ])
    cleaned = remove_obvious_noise(nodes)
    assert list(cleaned["text_context"]) == ["Sacramento College Fair"]


def test_cluster_candidates_end_to_end(node_factory):
    # Two events far apart; expansion pool == candidates here.
    rows = (
        [{"rendering_order": i, "parent_index": 1, "contains_date": int(i == 0)} for i in range(3)]
        + [{"rendering_order": 60 + i, "parent_index": 50, "contains_date": int(i == 0)} for i in range(3)]
    )
    nodes = node_factory(rows)
    clustered = cluster_candidates(nodes, nodes)
    assert clustered["pred_cluster"].nunique() == 2
    assert len(clustered) == 6  # dedup kept every node exactly once
