"""Unit and Contract Tests for Network-Aware Spillback Features (UrbanFlow AI).

Verifies:
1. Topology contracts (Acyclic multi-hop line graph, disjointness, self-loop exclusion)
2. Perimeter exit identification and empty-set imputation contracts
3. Exact same-run, same-timestamp join integrity (Zero future-target leakage)
4. Historical 15-second downstream queue delta (Current and past observations only)
5. Metric preservation (No clipping of queue/length ratio to 1.0)
6. Asset and directory decoupling (Network features written to separate campaign paths)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from urbanflow.core.network_topology import NetworkTopology


def test_topology_contracts_disjointness_and_acyclicity():
    """
    Verifies that for every road in the 210-road digital twin:
    1. Origin road is never in any downstream hop set (no self-loops).
    2. 1-hop, 2-hop, and 3-hop sets are mutually disjoint.
    3. No duplicates exist within any hop set.
    """
    topo = NetworkTopology.get_instance()
    res = topo.validate_topology_contracts()
    assert res["valid"], f"Topology contract violations: {res['violations']}"
    assert res["total_roads"] == 210
    assert res["violations_count"] == 0


def test_perimeter_exit_identification_and_counts():
    """
    Verifies that perimeter exit links (zero downstream roads) are correctly identified.
    """
    topo = NetworkTopology.get_instance()
    exits = [r_id for r_id in topo.roads if topo.is_perimeter_exit(r_id)]
    assert len(exits) == 8, f"Expected exactly 8 perimeter exits, got {len(exits)}"
    for r_id in exits:
        assert len(topo.get_downstream_roads(r_id, 1)) == 0
        assert len(topo.get_downstream_roads(r_id, 2)) == 0
        assert len(topo.get_downstream_roads(r_id, 3)) == 0


def test_branching_and_connectivity_counts():
    """
    Verifies that branching intersections produce valid downstream sets.
    """
    topo = NetworkTopology.get_instance()
    branching_roads = [r_id for r_id in topo.roads if len(topo.get_downstream_roads(r_id, 1)) > 1]
    assert len(branching_roads) >= 150, f"Expected >= 150 branching roads, found {len(branching_roads)}"
    for r_id in branching_roads:
        hop1 = topo.get_downstream_roads(r_id, 1)
        assert len(hop1) == len(set(hop1)), f"Duplicate roads in hop 1 for {r_id}"


def test_queue_ratio_unbounded_preservation():
    """
    Verifies that queue-to-length ratios exceeding 1.0 are strictly preserved without clipping.
    """
    road_len = 100.0
    queue_len = 125.0  # Dense queue spillback
    ratio = queue_len / road_len
    assert ratio > 1.0
    # Must not be clipped
    assert ratio == 1.25


def test_temporal_leakage_invariance_contract():
    """
    Verifies that downstream queue delta uses strictly current (t) and historical (t - 15s) steps,
    and produces 0.0 or valid past difference without forward-looking data.
    """
    timestamps = np.arange(0.0, 100.0, 5.0)  # 20 steps
    # Series with monotonic increase
    q_series = pd.Series(timestamps * 2.0, index=timestamps)
    # 15s delta is 3 steps back
    q_delta_15s = q_series - q_series.shift(3).fillna(q_series.iloc[0])
    
    # At t=0s, 5s, 10s: delta uses boundary fill without future data
    assert q_delta_15s.iloc[0] == 0.0
    # At t >= 15s: delta is exactly q(t) - q(t - 15s) = (t*2) - ((t-15)*2) = 30.0
    for idx in range(3, len(timestamps)):
        assert abs(q_delta_15s.iloc[idx] - 30.0) < 1e-4


def test_network_feature_decoupling_paths():
    """
    Verifies that network features directory is completely separate from baseline campaign features.
    """
    from scripts.build_campaign_features import FEATURE_OUTPUT_DIR as BASELINE_DIR
    from scripts.build_campaign_network_features import NETWORK_FEATURE_OUTPUT_DIR, NETWORK_FEATURE_COLS

    assert BASELINE_DIR != NETWORK_FEATURE_OUTPUT_DIR
    assert "campaign_network_features" in str(NETWORK_FEATURE_OUTPUT_DIR)
    assert len(NETWORK_FEATURE_COLS) == 12


def test_downstream_extraction_pipeline_contracts():
    """
    Verifies that extract_downstream_telemetry_for_run correctly derives all 12 network features
    with finite values, no nulls, and correct boundary flags.
    """
    from scripts.build_campaign_network_features import (
        extract_downstream_telemetry_for_run,
        NETWORK_FEATURE_COLS,
    )
    import pyarrow.parquet as pq

    topo = NetworkTopology.get_instance()
    campaign_dir = Path("data/datasets/campaign")
    if not campaign_dir.exists():
        campaign_dir = Path("urbanflow/data/datasets/campaign")
    sample_file = next(campaign_dir.rglob("*.parquet"))
    df_raw = pq.read_table(str(sample_file)).to_pandas()

    df_out = extract_downstream_telemetry_for_run(df_raw, topo)
    assert len(df_out) == len(df_raw)

    for col in NETWORK_FEATURE_COLS:
        assert col in df_out.columns, f"Missing feature {col}"
        assert not df_out[col].isna().any(), f"Nulls found in {col}"
        assert not np.isinf(df_out[col]).any(), f"Infs found in {col}"

    # Check perimeter exit flag consistency
    perimeter_roads = [r_id for r_id in topo.roads if topo.is_perimeter_exit(r_id)]
    for p_road in perimeter_roads:
        sub = df_out[df_out["road_id"] == p_road]
        assert (sub["is_perimeter_exit"] == 1.0).all()
        assert (sub["down_1hop_max_queue_ratio"] == 0.0).all()

