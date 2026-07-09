"""
EDA ciblé : gaps de rendering_order intra-event vs inter-event,
groupés par parent_index.
"""
import pandas as pd
from dateclf.data import load_config, load_raw_merged, add_is_event_member_label

cfg = load_config("../config.yaml")
df = load_raw_merged(cfg)
df = add_is_event_member_label(df, cfg)

annotated = df[df['event_id'].notna()].copy()

for site_id, site_df in annotated.groupby('site_id'):
    site_df = site_df.sort_values('rendering_order')
    
    # Pour chaque événement, quel est le rendering_order de chaque parent_index ?
    # Et le gap entre parents consécutifs du même événement ?
    intra_gaps = []
    for eid, edf in site_df.groupby('event_id'):
        parents = edf.groupby('parent_index')['rendering_order'].min().sort_values()
        if len(parents) > 1:
            gaps = parents.diff().dropna().values
            intra_gaps.extend(gaps)
    
    # Gap entre le dernier node d'un événement et le premier du suivant
    inter_gaps = []
    event_bounds = site_df.groupby('event_id')['rendering_order'].agg(['min', 'max'])
    event_bounds = event_bounds.sort_values('min')
    for i in range(len(event_bounds) - 1):
        gap = event_bounds.iloc[i+1]['min'] - event_bounds.iloc[i]['max']
        inter_gaps.append(gap)
    
    if not intra_gaps and not inter_gaps:
        continue
    
    print(f"\n--- {site_id} ({site_df['event_id'].nunique()} events) ---")
    if intra_gaps:
        s = pd.Series(intra_gaps)
        print(f"  Intra-event parent gaps  — mean: {s.mean():.1f}, "
              f"min: {s.min():.0f}, max: {s.max():.0f}, p90: {s.quantile(.9):.0f}")
    if inter_gaps:
        s = pd.Series(inter_gaps)
        print(f"  Inter-event gaps         — mean: {s.mean():.1f}, "
              f"min: {s.min():.0f}, max: {s.max():.0f}, p10: {s.quantile(.1):.0f}")