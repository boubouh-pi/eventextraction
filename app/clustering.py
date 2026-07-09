"""
Inference-safe DOM clustering.

Ported from ``src/dateclf/train.py`` with one crucial change: the training
version's cluster-expansion step used the ground-truth ``label`` column to
pull nearby Date/Time/Location nodes into a cluster. At inference there are
no labels, so expansion relies on the *pattern* features
(``contains_date`` / ``contains_time`` / ``text_word_location``) instead.

The four post-processing steps and their order are identical to the
validated training pipeline:
  1. expand clusters with nearby short field nodes missed by top-K
  2. split mega-clusters at event boundaries (repeating date/time pattern)
  3. deduplicate nodes copied into multiple clusters by expansion
  4. merge tiny fragments into their nearest neighbour
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def unified_clustering(
    nodes: pd.DataFrame,
    gap_order: int = 10,
    gap_parent: int = 8,
    gap_depth: int = 2,
) -> pd.DataFrame:
    """Sequential clustering on rendering_order + parent_index + depth."""
    nodes = nodes.sort_values("rendering_order").reset_index(drop=True).copy()

    cluster_ids = []
    current_cluster = 0

    for i, row in nodes.iterrows():
        if i == 0:
            current_cluster += 1
        else:
            prev = nodes.iloc[i - 1]
            gap_r = row["rendering_order"] - prev["rendering_order"]
            if pd.notna(row["parent_index"]) and pd.notna(prev["parent_index"]):
                gap_p = abs(row["parent_index"] - prev["parent_index"])
            else:
                gap_p = 0
            gap_d = abs(row["depth"] - prev["depth"])

            if gap_r > gap_order and (gap_p > gap_parent or gap_d > gap_depth):
                current_cluster += 1

        cluster_ids.append(current_cluster)

    nodes["pred_cluster"] = cluster_ids
    return nodes


def expand_clusters_with_nearby_fields(
    clustered: pd.DataFrame,
    all_scored_nodes: pd.DataFrame,
    window: int = 8,
) -> pd.DataFrame:
    """
    Pull short field-like nodes ("7", "mar", "thu") that scored below top-K
    but sit inside a cluster's rendering window. Field-likeness comes from
    pattern features, never from labels.
    """
    expanded = clustered.copy()
    already_used = set(clustered["row_id"]) if "row_id" in clustered.columns else set()

    for cluster_id in clustered["pred_cluster"].unique():
        cluster_nodes = clustered[clustered["pred_cluster"] == cluster_id]
        min_order = cluster_nodes["rendering_order"].min()
        max_order = cluster_nodes["rendering_order"].max()

        nearby = all_scored_nodes[
            (all_scored_nodes["rendering_order"] >= min_order - window)
            & (all_scored_nodes["rendering_order"] <= max_order + window)
        ].copy()
        if "row_id" in nearby.columns:
            nearby = nearby[~nearby["row_id"].isin(already_used)]

        if nearby.empty:
            continue

        field_like = (
            nearby.get("contains_date", 0).astype(bool)
            | nearby.get("contains_time", 0).astype(bool)
            | nearby.get("text_word_location", 0).astype(bool)
        )
        field_nodes = nearby[field_like]

        if not field_nodes.empty:
            field_nodes = field_nodes.copy()
            field_nodes["pred_cluster"] = cluster_id
            expanded = pd.concat([expanded, field_nodes], ignore_index=True)
            if "row_id" in field_nodes.columns:
                already_used.update(field_nodes["row_id"])

    return expanded.sort_values("rendering_order").reset_index(drop=True)


def sub_cluster_by_event_boundaries(clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Split mega-clusters at event boundaries: inside a dense listing, each
    date-like node following non-date content starts a new event.
    """
    result = clustered.copy()
    max_existing = int(result["pred_cluster"].max()) + 1
    offset = max_existing * 1000

    for cluster_id in clustered["pred_cluster"].unique():
        mask = result["pred_cluster"] == cluster_id
        cluster = result[mask].sort_values("rendering_order")

        if len(cluster) <= 10:
            continue
        if "contains_date" not in cluster.columns and "contains_time" not in cluster.columns:
            continue

        n_date_nodes = int(cluster.get("contains_date", pd.Series(dtype=int)).sum())
        n_time_nodes = int(cluster.get("contains_time", pd.Series(dtype=int)).sum())

        # <3 temporal markers usually means one event with duplicated fields.
        if n_date_nodes < 3 and n_time_nodes < 3:
            continue

        use_date = n_date_nodes >= 2

        sub_id = 0
        prev_was_boundary = True
        sub_ids = []
        for _, row in cluster.iterrows():
            is_boundary = bool(row.get("contains_date" if use_date else "contains_time", False))
            if is_boundary and not prev_was_boundary and len(sub_ids) > 0:
                sub_id += 1
            sub_ids.append(sub_id)
            prev_was_boundary = is_boundary

        if sub_id >= 1:
            new_ids = [offset + cluster_id * 100 + s for s in sub_ids]
            result.loc[cluster.index, "pred_cluster"] = new_ids

    return result


def deduplicate_expanded_nodes(clustered: pd.DataFrame) -> pd.DataFrame:
    """Keep each row_id only in the cluster whose median order it is closest to."""
    if "row_id" not in clustered.columns:
        return clustered

    result = clustered.copy()

    for row_id, group in result.groupby("row_id"):
        if len(group) <= 1 or group["pred_cluster"].nunique() <= 1:
            continue

        best_idx, best_dist = None, float("inf")
        for idx, row in group.iterrows():
            cluster_nodes = result[result["pred_cluster"] == row["pred_cluster"]]
            dist = abs(row["rendering_order"] - cluster_nodes["rendering_order"].median())
            if dist < best_dist:
                best_dist, best_idx = dist, idx

        result = result.drop([i for i in group.index if i != best_idx])

    return result.reset_index(drop=True)


def merge_tiny_subclusters(clustered: pd.DataFrame, min_size: int = 3) -> pd.DataFrame:
    """
    Merge fragments below ``min_size`` into their nearest cluster, unless the
    merge would combine >2 date markers (two distinct events).
    """
    result = clustered.copy()
    cluster_sizes = result.groupby("pred_cluster").size()
    tiny_clusters = cluster_sizes[cluster_sizes < min_size].index.tolist()

    for tiny_id in tiny_clusters:
        tiny_mask = result["pred_cluster"] == tiny_id
        if tiny_mask.sum() == 0:
            continue

        tiny_nodes = result[tiny_mask]
        tiny_center = tiny_nodes["rendering_order"].median()
        other_clusters = result[~tiny_mask].groupby("pred_cluster")["rendering_order"].median()
        if other_clusters.empty:
            continue

        nearest_cluster = (other_clusters - tiny_center).abs().idxmin()

        if "contains_date" in result.columns:
            nearest_nodes = result[result["pred_cluster"] == nearest_cluster]
            combined_dates = int(tiny_nodes["contains_date"].sum()) + int(
                nearest_nodes["contains_date"].sum()
            )
            if combined_dates > 2:
                continue

        result.loc[tiny_mask, "pred_cluster"] = nearest_cluster

    return result


def remove_obvious_noise(nodes: pd.DataFrame) -> pd.DataFrame:
    """Post-scoring filter for navigation/footer/header noise."""
    nodes = nodes.copy()
    text = nodes["text_context"].fillna("").astype(str).str.lower().str.strip()

    noise_patterns = text.str.contains(
        r"\b(?:login|log in|sign in|register|home|menu|navigation|"
        r"contact us|about us|privacy|copyright|call us|"
        r"step 1|step 2|pricing|membership)\b",
        regex=True,
        na=False,
    )
    structural_noise = (nodes["depth"] < 2) | (nodes["text_length"] < 5)

    return nodes[~(noise_patterns | structural_noise)].copy()


# ---------------------------------------------------------------------------
# Periodic clustering
#
# Diagnosis of the 0%-reconstitution sites (sacac, wacac, outofstate,
# neacac_spring): event listings are STRICTLY PERIODIC — the same sequence
# of DOM units repeats once per event — and the rendering gap *between*
# events equals the gap *inside* an event, so gap-based clustering cannot
# separate them. Meanwhile parent groups are almost never impure (6 of
# ~700 across all 16 sites mix two events).
#
# Algorithm:
#   1. atomic units = parent groups of the candidates, in rendering order
#   2. unit signature = (sorted tags, has_date, has_time, has_heading)
#   3. period P = smallest shift maximizing signature autocorrelation
#   4. marker = once-per-period signature with the most regular spacing
#   5. boundary offset = the cut alignment maximizing rendering gaps at cuts
#   6. segment ranges partition the page; every node (candidate or not)
#      falling in a segment's range joins that event → recovers short
#      low-scoring field nodes ("thu", "january", "29") without labels
# Returns None when no reliable period exists; caller falls back to the
# gap-based chain.
# ---------------------------------------------------------------------------

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

_NAV_NOISE_RE = (
    r"\b(?:login|log in|sign in|register|home|menu|navigation|"
    r"contact us|about us|privacy|copyright|call us|"
    r"step 1|step 2|pricing|membership)\b"
)


def _build_units(nodes: pd.DataFrame) -> list[dict]:
    units = []
    for pid, grp in nodes.groupby("parent_index", dropna=False, sort=False):
        if pd.isna(pid):
            # unknown parent: never merge — one unit per node
            for idx, row in grp.iterrows():
                units.append(_unit_from(grp.loc[[idx]]))
        else:
            units.append(_unit_from(grp))
    units.sort(key=lambda u: u["min_order"])
    return units


def _unit_from(grp: pd.DataFrame) -> dict:
    tags = tuple(sorted(str(t).lower() for t in grp["tag"]))
    ordered = grp.sort_values("rendering_order")
    return {
        "idx": grp.index,
        "min_order": float(grp["rendering_order"].min()),
        "max_order": float(grp["rendering_order"].max()),
        "first_depth": float(ordered["depth"].iloc[0]),
        "last_depth": float(ordered["depth"].iloc[-1]),
        "sig": (
            tags,
            bool(grp["contains_date"].any()),
            bool(grp.get("contains_time", pd.Series(False, index=grp.index)).any()),
            any(t in _HEADING_TAGS for t in tags),
        ),
    }


def _find_marker(units: list[dict]) -> tuple[list[int], int] | None:
    """
    Find a once-per-event marker: a family of units appearing ≥3 times with
    regular spacing and covering most of the listing.

    Families are exact signatures PLUS the pattern flags (has_date /
    has_time / has_heading) on their own — sites like neacac mix markup
    between events (date in <td> on one row, <p> on the next), so the flag
    survives where the exact signature does not.

    Returns (marker unit indices, period in units) or None.
    """
    m = len(units)
    families: dict = {}
    for i, u in enumerate(units):
        families.setdefault(("sig", u["sig"]), []).append(i)
        tags, has_date, has_time, has_heading = u["sig"]
        if has_date:
            families.setdefault(("flag", "date"), []).append(i)
        if has_time:
            families.setdefault(("flag", "time"), []).append(i)
        if has_heading:
            families.setdefault(("flag", "heading"), []).append(i)

    # Qualify families on their REGULAR CORE: the longest run of consecutive
    # occurrences with homogeneous spacing. Live candidate pools interleave
    # noise units (nav links share the ('a',) signature with event name
    # links), which would explode a naive whole-family CV; the core trims
    # those irregular prefixes/suffixes away.
    qualified: list[tuple[list[int], int]] = []
    for _, pos in families.items():
        core = _regular_core(pos)
        # ≥4 occurrences: three evenly-spaced units happen by coincidence
        # on small pages, and a false period costs more than the gap-chain
        # fallback it would replace
        if len(core) < 4:
            continue
        gaps = np.diff(core)
        mean_gap = float(np.mean(gaps))
        if mean_gap < 1.0:
            continue
        cv = float(np.std(gaps)) / mean_gap
        coverage = (core[-1] - core[0] + mean_gap) / m
        if cv > 0.30 or coverage < 0.50:
            continue
        qualified.append((core, max(1, round(mean_gap))))
    if not qualified:
        return None

    # Pages nest several periodicities (sections ⊃ events ⊃ fields), so
    # several families qualify. The event-level one is the family whose
    # segments are structurally SIMILAR to each other and of balanced size
    # (the definition of a periodic listing): over-segmentation cuts inside
    # events → consecutive segments differ; under-segmentation also scores
    # high on similarity, so near-ties resolve to the FINEST segmentation.
    def segmentation_quality(pos: list[int], period: int) -> float:
        cuts = _clip_cuts(_pick_offset(units, pos, period, m), m)
        if not cuts:
            return 0.0
        bounds = ([0] if cuts[0] != 0 else []) + cuts + [m]
        segments = [
            Counter(u["sig"] for u in units[bounds[i]:bounds[i + 1]])
            for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]
        ]
        if len(segments) < 2:
            return 0.0
        sims = []
        for a, b in zip(segments, segments[1:]):
            inter = sum((a & b).values())
            union = sum((a | b).values())
            sims.append(inter / union if union else 0.0)
        sizes = [sum(s.values()) for s in segments]
        balance = 1.0 / (1.0 + float(np.std(sizes)) / max(float(np.mean(sizes)), 1e-9))
        return float(np.mean(sims)) * balance

    scored = [
        (segmentation_quality(pos, period), len(pos), pos, period)
        for pos, period in qualified
    ]
    best_quality = max(q for q, _, _, _ in scored)
    if best_quality <= 0.0:
        return None
    # Coarse (section-level) segmentations saturate similarity×balance, so
    # the window is wide and the FINEST segmentation inside it wins; truly
    # over-fine families collapse the similarity term and fall out of it.
    near_ties = [t for t in scored if t[0] >= 0.70 * best_quality]
    _, _, pos, period = max(near_ties, key=lambda t: t[1])
    return pos, period


def _regular_core(pos: list[int]) -> list[int]:
    """Longest run of consecutive occurrences with spacing within 50% of
    the median spacing — trims irregular noise at either end of a family."""
    if len(pos) < 3:
        return pos
    gaps = np.diff(pos)
    med = float(np.median(gaps))
    if med <= 0:
        return pos
    ok = (gaps >= 0.5 * med) & (gaps <= 1.5 * med)

    best_start, best_len = 0, 0
    run_start, run_len = 0, 0
    for i, good in enumerate(ok):
        if good:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    if best_len == 0:
        return []
    # a run of best_len gaps spans best_len + 1 occurrences
    return list(pos[best_start:best_start + best_len + 1])


def _cuts_for(pos: list[int], offset: int) -> list[int]:
    return sorted({max(0, p - offset) for p in pos})


def _clip_cuts(cuts: list[int], m: int) -> list[int]:
    return [c for c in cuts if c < m]


def _mean_cut_gap(units: list[dict], pos: list[int], offset: int) -> float:
    """
    MEDIAN rendering gap at the cuts (cut at 0 carries no gap signal).
    Median, not mean: candidate pools sample the page unevenly, and a few
    cuts landing after missing units create outlier gaps that would make
    a systematically-wrong alignment (gap 1 at most cuts) look good.
    """
    cuts = [c for c in _cuts_for(pos, offset) if 0 < c < len(units)]
    if not cuts:
        return 0.0
    return float(np.median(
        [units[c]["min_order"] - units[c - 1]["max_order"] for c in cuts]
    ))


def _rotation_prior(units: list[dict], cuts: list[int], m: int) -> float:
    """
    Semantic phase disambiguation. A periodic listing is rotation-invariant
    structurally (any [ABCDE][ABCDE]… cut is self-consistent), and the
    rendering gap often peaks before the *description/date-tail* block
    (deeper subtree), not before the true event start. But semantically an
    event STARTS with its heading/name and its date comes early — never
    last. Bonus grows as the first heading (fallback: date, then time)
    unit sits earlier in each segment.
    """
    n_targets = max(3, len(cuts) // 2)
    for key in ("heading", "date", "time"):
        flag_idx = {"date": 1, "time": 2, "heading": 3}[key]
        # the key must recur about once per segment — a single stray
        # heading elsewhere on the page must not hijack the prior
        if sum(bool(u["sig"][flag_idx]) for u in units) >= n_targets:
            break
    else:
        return 0.0

    cuts = _clip_cuts(list(cuts), m)
    if not cuts:
        return 0.0
    bounds = ([0] if cuts[0] != 0 else []) + cuts + [m]
    scores = []
    for i in range(len(bounds) - 1):
        seg = units[bounds[i]:bounds[i + 1]]
        if not seg:
            continue
        first = next(
            (j for j, u in enumerate(seg) if u["sig"][flag_idx]), len(seg)
        )
        # saturating: "in the first half" is all that matters — the name
        # may legitimately precede the date, so position 1 is as good as 0
        scores.append(1.0 if first / max(len(seg), 1) <= 0.5 else 0.0)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _pick_offset(units: list[dict], pos: list[int], period: int, m: int) -> list[int]:
    """
    Best boundary alignment: rendering gap and rotation prior each
    normalised to [0, 1] and equally weighted — raw gaps have arbitrary
    per-page scale and would otherwise drown the semantic signal.
    """
    # offset -1 covers markers that END their event (trailing description
    # unit): the cut then falls right after each marker occurrence.
    offsets = range(-1, period)
    gaps = {o: _mean_cut_gap(units, pos, o) for o in offsets}
    max_gap = max(gaps.values()) or 1.0
    best_offset = max(
        offsets,
        key=lambda o: gaps[o] / max_gap
        + _rotation_prior(units, _cuts_for(pos, o), m),
    )
    return _cuts_for(pos, best_offset)


def _segment_recursively(units: list[dict], depth: int = 0) -> list[list[dict]]:
    """
    Segment a unit sequence, then recurse into large segments: pages nest
    periodicities (page ⊃ sections ⊃ events), and the best global marker is
    often the *section* one — each section then exposes a clean event-level
    period of its own. Small segments (<8 units, i.e. single events) stop
    the recursion because their marker families never reach 3 occurrences.
    """
    if len(units) < 8 or depth >= 3:
        return [units]
    found = _find_marker(units)
    if found is None:
        return [units]
    pos, period = found
    cuts = _clip_cuts(_pick_offset(units, pos, period, len(units)), len(units))
    if not cuts:
        return [units]
    bounds = ([0] if cuts[0] != 0 else []) + cuts + [len(units)]
    segments = [
        units[bounds[i]:bounds[i + 1]]
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]
    if len(segments) <= 1:
        return [units]
    out: list[list[dict]] = []
    for seg in segments:
        out.extend(_segment_recursively(seg, depth + 1))
    return out


def _periodic_candidate_pool(
    top_k: pd.DataFrame, all_scored_nodes: pd.DataFrame
) -> pd.DataFrame:
    """
    Periodicity detection needs structural completeness, not high-precision
    candidates: the denoised top-K drops the short field nodes ("07",
    "mar", "9am") whose units carry the repeating pattern, leaving holes
    that break marker spacing. Rebuild a wider pool by score — roughly
    twice the top-K, capped — filtered only for nav text and shallow
    chrome, never for length.
    """
    if "score" not in all_scored_nodes.columns:
        return top_k
    n = min(len(all_scored_nodes), max(2 * len(top_k), 60), 400)
    pool = all_scored_nodes.nlargest(n, "score").copy()
    text = pool["text_context"].fillna("").astype(str).str.lower()
    pool = pool[~text.str.contains(_NAV_NOISE_RE, regex=True, na=False)]
    if "depth" in pool.columns:
        pool = pool[pool["depth"] >= 2]
    return pool if len(pool) >= 2 else top_k


def periodic_clustering(
    candidates: pd.DataFrame,
    all_scored_nodes: pd.DataFrame,
) -> pd.DataFrame | None:
    nodes = candidates.sort_values("rendering_order").copy()
    units = _build_units(nodes)
    if len(units) < 8:
        return None

    segments = _segment_recursively(units)
    if len(segments) <= 1:
        return None

    # Sanity check: an event listing carries temporal signals. When most
    # leaf segments have no date/time/heading unit, the detected period is
    # page chrome (nav menus repeat too) — bail out to the gap chain.
    flagged = sum(
        any(u["sig"][1] or u["sig"][2] or u["sig"][3] for u in seg)
        for seg in segments
    )
    if flagged / len(segments) < 0.5:
        return None

    result = nodes.copy()
    result["pred_cluster"] = 0
    ranges: list[tuple[float, float, int]] = []
    for seg_id, seg_units in enumerate(segments):
        for u in seg_units:
            result.loc[u["idx"], "pred_cluster"] = seg_id
        start = seg_units[0]["min_order"]
        end = (segments[seg_id + 1][0]["min_order"]
               if seg_id + 1 < len(segments) else float("inf"))
        ranges.append((start, end, seg_id))

    # segment-range expansion: recover non-candidate nodes inside each
    # segment's span (short date fragments score poorly but belong here)
    used = set(result["row_id"]) if "row_id" in result.columns else set()
    extra_frames = []
    pool = all_scored_nodes
    if "row_id" in pool.columns:
        pool = pool[~pool["row_id"].isin(used)]
    text = pool["text_context"].fillna("").astype(str).str.lower()
    pool = pool[~text.str.contains(_NAV_NOISE_RE, regex=True, na=False)]
    for start, end, seg_id in ranges:
        inside = pool[(pool["rendering_order"] >= start) & (pool["rendering_order"] < end)]
        if not inside.empty:
            inside = inside.copy()
            inside["pred_cluster"] = seg_id
            extra_frames.append(inside)
            if "row_id" in inside.columns:
                pool = pool[~pool["row_id"].isin(inside["row_id"])]

    if extra_frames:
        result = pd.concat([result] + extra_frames, ignore_index=True)
    result["cluster_method"] = "periodic"
    return result.sort_values("rendering_order").reset_index(drop=True)


def cluster_candidates(
    top_k: pd.DataFrame,
    all_scored_nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clustering on denoised top-K candidates: periodic clustering when the
    page is a repeating listing, else the validated gap-based chain.
    Returns the input with a ``pred_cluster`` column.
    """
    if len(top_k) < 2:
        out = top_k.copy()
        out["pred_cluster"] = 1
        return out

    periodic = periodic_clustering(
        _periodic_candidate_pool(top_k, all_scored_nodes), all_scored_nodes
    )
    if periodic is not None:
        return periodic

    clustered = unified_clustering(top_k, gap_order=10, gap_parent=8, gap_depth=2)

    # Retry tighter when one cluster swallowed the page.
    if clustered.groupby("pred_cluster").size().max() > 15:
        clustered = unified_clustering(top_k, gap_order=5, gap_parent=4, gap_depth=1)

    clustered = expand_clusters_with_nearby_fields(clustered, all_scored_nodes, window=8)
    clustered = sub_cluster_by_event_boundaries(clustered)
    clustered = deduplicate_expanded_nodes(clustered)
    clustered = merge_tiny_subclusters(clustered, min_size=3)
    clustered["cluster_method"] = "gaps"
    return clustered
