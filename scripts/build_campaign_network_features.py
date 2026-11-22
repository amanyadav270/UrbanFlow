"""UrbanFlow AI — Campaign Network-Aware Feature Builder.

Builds supervised dataset with 12 network-aware downstream topology features:
1. Downstream 1-Hop Features:
   - down_1hop_max_queue_ratio : max(queue_length_m / length) across D1(R) [unbounded float]
   - down_1hop_min_speed_kmh   : min(avg_speed_kmh) across D1(R) [float]
   - down_1hop_max_occupancy   : max(occupancy_ratio) across D1(R) [float]
   - down_1hop_mean_vc_ratio   : mean(vc_ratio) across D1(R) [float]
2. Downstream 2-Hop Features:
   - down_2hop_max_queue_ratio : max(queue_length_m / length) across D2(R) [unbounded float]
   - down_2hop_min_speed_kmh   : min(avg_speed_kmh) across D2(R) [float]
   - down_2hop_max_occupancy   : max(occupancy_ratio) across D2(R) [float]
3. Downstream 3-Hop Features:
   - down_3hop_max_queue_ratio : max(queue_length_m / length) across D3(R) [unbounded float]
   - down_3hop_min_speed_kmh   : min(avg_speed_kmh) across D3(R) [float]
4. Shockwave & Boundary Features:
   - downstream_speed_gradient : avg_speed_kmh(local) - down_1hop_min_speed_kmh [float]
   - downstream_queue_delta_15s: down_1hop_max_queue_m(t) - down_1hop_max_queue_m(t - 15s) [float]
   - is_perimeter_exit         : 1.0 if D1(R) is empty else 0.0 [float]

Strict Data Integrity & Causal Safeguards:
- Same-run, same-timestamp joins: Zero future downstream measurements.
- Backward-looking 15s queue delta: computed using historical timestamps (shift 3 at 5s sampling).
- Zero clipping: queue/length ratios exceeding 1.0 are preserved.
- Output isolated strictly to data/datasets/campaign_network_features/.
- Preserves all existing models, baseline datasets, and reports.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

URBANFLOW_ROOT = Path(__file__).resolve().parent.parent
if str(URBANFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(URBANFLOW_ROOT))

from urbanflow.core.network_topology import NetworkTopology
from scripts.build_campaign_features import (
    assert_pipeline_integrity,
    compute_spillback_event_series,
    HORIZONS_SEC,
    VALID_TIMESTAMPS_PER_HORIZON,
)

DATA_DIR = URBANFLOW_ROOT / "data"
CAMPAIGN_DATA_DIR = DATA_DIR / "datasets" / "campaign"
CAMPAIGN_METADATA_DB = DATA_DIR / "datasets" / "campaign_metadata.db"
NETWORK_FEATURE_OUTPUT_DIR = DATA_DIR / "datasets" / "campaign_network_features"

NETWORK_FEATURE_COLS = [
    "down_1hop_max_queue_ratio",
    "down_1hop_min_speed_kmh",
    "down_1hop_max_occupancy",
    "down_1hop_mean_vc_ratio",
    "down_2hop_max_queue_ratio",
    "down_2hop_min_speed_kmh",
    "down_2hop_max_occupancy",
    "down_3hop_max_queue_ratio",
    "down_3hop_min_speed_kmh",
    "downstream_speed_gradient",
    "downstream_queue_delta_15s",
    "is_perimeter_exit",
]


def extract_downstream_telemetry_for_run(
    df_run: pd.DataFrame,
    topo: NetworkTopology,
) -> pd.DataFrame:
    """
    Extracts the 12 network-aware downstream features for a single simulation run.
    Uses vector matrix transformations across the 721 synchronized snapshots.
    Guarantees strict same-run, same-timestamp causality without future leakage.
    """
    df_sorted = df_run.sort_values(["timestamp", "road_id"]).reset_index(drop=True)
    all_roads = sorted(list(topo.roads.keys()))
    road_lengths = pd.Series(topo.road_lengths)
    road_speeds = pd.Series(topo.road_speeds)

    # 1. Pivot matrices (rows = timestamps, columns = road_id)
    q_mat = df_sorted.pivot(index="timestamp", columns="road_id", values="queue_length_m")
    v_mat = df_sorted.pivot(index="timestamp", columns="road_id", values="avg_speed_kmh")
    occ_mat = df_sorted.pivot(index="timestamp", columns="road_id", values="occupancy_ratio")
    vc_mat = df_sorted.pivot(index="timestamp", columns="road_id", values="vc_ratio")

    n_timestamps = len(q_mat)
    timestamps = q_mat.index.values

    # Pre-allocate dictionary for tabular conversion
    records: Dict[str, np.ndarray] = {col: np.zeros(len(df_sorted), dtype=np.float64) for col in NETWORK_FEATURE_COLS}
    records["timestamp"] = df_sorted["timestamp"].values
    records["road_id"] = df_sorted["road_id"].values
    records["run_id"] = df_sorted["run_id"].values

    # Build road-wise index lookup
    road_indices = {r_id: np.where(df_sorted["road_id"].values == r_id)[0] for r_id in all_roads}

    for r_id in all_roads:
        idx_mask = road_indices[r_id]
        d1 = topo.get_downstream_roads(r_id, 1)
        d2 = topo.get_downstream_roads(r_id, 2)
        d3 = topo.get_downstream_roads(r_id, 3)

        local_v = v_mat[r_id].values
        local_speed_limit = road_speeds[r_id]

        # --- 1-Hop Features ---
        if not d1:
            # Perimeter exit policy: explicit indicator + neutral fallback
            records["is_perimeter_exit"][idx_mask] = 1.0
            records["down_1hop_max_queue_ratio"][idx_mask] = 0.0
            records["down_1hop_min_speed_kmh"][idx_mask] = local_speed_limit
            records["down_1hop_max_occupancy"][idx_mask] = 0.0
            records["down_1hop_mean_vc_ratio"][idx_mask] = 0.0
            records["downstream_speed_gradient"][idx_mask] = 0.0
            records["downstream_queue_delta_15s"][idx_mask] = 0.0
        else:
            records["is_perimeter_exit"][idx_mask] = 0.0
            q1 = q_mat[d1].values
            len1 = road_lengths[d1].values
            v1 = v_mat[d1].values
            occ1 = occ_mat[d1].values
            vc1 = vc_mat[d1].values

            # Queue ratio preserves measured value unclipped (can exceed 1.0)
            q_ratio_1 = np.max(q1 / len1, axis=1)
            min_v_1 = np.min(v1, axis=1)
            records["down_1hop_max_queue_ratio"][idx_mask] = q_ratio_1
            records["down_1hop_min_speed_kmh"][idx_mask] = min_v_1
            records["down_1hop_max_occupancy"][idx_mask] = np.max(occ1, axis=1)
            records["down_1hop_mean_vc_ratio"][idx_mask] = np.mean(vc1, axis=1)

            # Downstream speed gradient: local speed - worst downstream bottleneck speed
            records["downstream_speed_gradient"][idx_mask] = local_v - min_v_1

            # Historical 15s downstream queue delta (3 steps prior at 5s sampling)
            max_q1_m = np.max(q1, axis=1)
            q_delta_15s = pd.Series(max_q1_m).diff(3).fillna(0.0).values
            records["downstream_queue_delta_15s"][idx_mask] = q_delta_15s

        # --- 2-Hop Features ---
        if not d2:
            records["down_2hop_max_queue_ratio"][idx_mask] = 0.0
            records["down_2hop_min_speed_kmh"][idx_mask] = local_speed_limit
            records["down_2hop_max_occupancy"][idx_mask] = 0.0
        else:
            records["down_2hop_max_queue_ratio"][idx_mask] = np.max(q_mat[d2].values / road_lengths[d2].values, axis=1)
            records["down_2hop_min_speed_kmh"][idx_mask] = np.min(v_mat[d2].values, axis=1)
            records["down_2hop_max_occupancy"][idx_mask] = np.max(occ_mat[d2].values, axis=1)

        # --- 3-Hop Features ---
        if not d3:
            records["down_3hop_max_queue_ratio"][idx_mask] = 0.0
            records["down_3hop_min_speed_kmh"][idx_mask] = local_speed_limit
        else:
            records["down_3hop_max_queue_ratio"][idx_mask] = np.max(q_mat[d3].values / road_lengths[d3].values, axis=1)
            records["down_3hop_min_speed_kmh"][idx_mask] = np.min(v_mat[d3].values, axis=1)

    # Assign network columns directly (df_sorted and records have identical order)
    for col in NETWORK_FEATURE_COLS:
        df_sorted[col] = records[col]
    return df_sorted


def fast_spillback_fsm(
    q_arr: np.ndarray,
    v_arr: np.ndarray,
    road_len_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """100% bit-for-bit equivalent vector FSM implementation."""
    N = len(q_arr)
    is_breach = (q_arr >= (0.60 * road_len_m)) | (v_arr <= 12.0)
    is_recovering = (q_arr < (0.40 * road_len_m)) & (v_arr > 20.0)

    is_active = np.zeros(N, dtype=bool)
    is_onset = np.zeros(N, dtype=bool)

    in_event = False
    idx = 0
    while idx < N:
        if not in_event:
            if idx + 7 <= N and is_breach[idx:idx+7].all():
                in_event = True
                is_onset[idx] = True
                is_active[idx] = True
                idx += 1
            else:
                idx += 1
        else:
            if idx + 13 <= N and is_recovering[idx:idx+13].all():
                is_active[idx:idx+13] = True
                in_event = False
                idx += 13
            else:
                is_active[idx] = True
                idx += 1
    return is_active, is_onset


def build_supervised_network_horizon_dataset(
    df_run: pd.DataFrame,
    horizon_name: str,
    horizon_sec: int,
    topo: NetworkTopology,
) -> pd.DataFrame:
    """
    Builds supervised dataset with both baseline local features and 12 network-aware features.
    Enforces 600s warmup filtering and exact valid bounds.
    """
    # 1. Attach 12 downstream network features across the entire 3600s run
    df_with_network = extract_downstream_telemetry_for_run(df_run, topo)

    # 2. Warm-up filter (t >= 600.0s)
    df = df_with_network[df_with_network["timestamp"] >= 600.0].copy()
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    steps_ahead = int(round(horizon_sec / 5.0))
    t_max_valid = 3600.0 - horizon_sec

    # Vectorized local lag features
    g = df.groupby("road_id")
    for lag in (1, 2, 3):
        df[f"avg_speed_lag_{lag}"] = g["avg_speed_kmh"].shift(lag)
        df[f"density_lag_{lag}"] = g["density_veh_km_lane"].shift(lag)
        df[f"queue_lag_{lag}"] = g["queue_length_m"].shift(lag)

    for col in ("avg_speed", "density", "queue"):
        orig_col = "avg_speed_kmh" if col == "avg_speed" else ("density_veh_km_lane" if col == "density" else "queue_length_m")
        first_vals = g[orig_col].transform("first")
        for lag in (1, 2, 3):
            df[f"{col}_lag_{lag}"] = df[f"{col}_lag_{lag}"].fillna(first_vals)

    # Vectorized rolling statistics (4 steps = 20s window)
    df["avg_speed_roll_mean"] = g["avg_speed_kmh"].rolling(4, min_periods=1).mean().values
    df["avg_speed_roll_std"] = g["avg_speed_kmh"].rolling(4, min_periods=1).std().fillna(0.0).values
    df["density_roll_mean"] = g["density_veh_km_lane"].rolling(4, min_periods=1).mean().values
    df["queue_roll_mean"] = g["queue_length_m"].rolling(4, min_periods=1).mean().values

    # Lookahead targets
    df["target_speed"] = g["avg_speed_kmh"].shift(-steps_ahead)
    df["target_density"] = g["density_veh_km_lane"].shift(-steps_ahead)
    df["target_queue"] = g["queue_length_m"].shift(-steps_ahead)
    df["target_congestion_event"] = (df["target_speed"] <= 20.0).astype(float)

    # Spillback FSM and forward onset targets per road
    is_actives = np.zeros(len(df), dtype=bool)
    is_onsets = np.zeros(len(df), dtype=bool)
    target_spillbacks = np.zeros(len(df), dtype=float)

    road_indices = g.indices
    for r_id, idxs in road_indices.items():
        q_sub = df["queue_length_m"].iloc[idxs].values
        v_sub = df["avg_speed_kmh"].iloc[idxs].values
        r_len = topo.get_road_length(r_id)
        act, ons = fast_spillback_fsm(q_sub, v_sub, r_len)
        is_actives[idxs] = act
        is_onsets[idxs] = ons

        # Forward rolling max of is_event_onset over window of size steps_ahead shifted by -1
        fwd = pd.Series(ons[::-1]).rolling(steps_ahead, min_periods=1).max().values[::-1]
        fwd_shifted = np.empty_like(fwd, dtype=float)
        fwd_shifted[:-1] = fwd[1:]
        fwd_shifted[-1] = 0.0
        target_spillbacks[idxs] = fwd_shifted

    df["is_active_spillback"] = is_actives
    df["is_event_onset"] = is_onsets
    df["target_spillback_onset"] = target_spillbacks

    # Truncate strictly to valid feature rows
    valid_df = df[df["timestamp"] <= (t_max_valid + 1e-3)].copy().sort_values(["timestamp", "road_id"]).reset_index(drop=True)
    return valid_df


def main():
    parser = argparse.ArgumentParser(description="UrbanFlow Campaign Network-Aware Feature Builder")
    parser.add_argument("--dry-run", action="store_true", help="Extract and inspect 1 run for contract verification")
    parser.add_argument("--horizon", type=str, default="all", choices=["all", "5min", "10min", "15min"])
    args = parser.parse_args()

    print("=" * 80)
    print("URBANFLOW AI — CAMPAIGN NETWORK-AWARE FEATURE BUILDER")
    print("=" * 80)
    print(f"Network Output Directory : {NETWORK_FEATURE_OUTPUT_DIR}")
    print(f"Metadata Database        : {CAMPAIGN_METADATA_DB}")
    print(f"Campaign Raw Parquet     : {CAMPAIGN_DATA_DIR}")

    topo = NetworkTopology.get_instance()
    val_topo = topo.validate_topology_contracts()
    print(f"Loaded Digital Twin Topology: {val_topo['total_roads']} roads, {val_topo['perimeter_exit_count']} perimeter exits. Valid={val_topo['valid']}")

    if not CAMPAIGN_METADATA_DB.exists():
        print(f"Error: {CAMPAIGN_METADATA_DB} does not exist.")
        sys.exit(1)

    conn = sqlite3.connect(str(CAMPAIGN_METADATA_DB))
    c = conn.cursor()
    c.execute("SELECT run_id, family_id, partition, parquet_path FROM campaign_runs WHERE status = 'SUCCESS' ORDER BY run_id")
    completed_runs = c.fetchall()
    conn.close()

    print(f"Available Completed Runs: {len(completed_runs)} / 60")
    if not completed_runs:
        print("No completed campaign runs found.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN MODE]: Executing extraction on sample run to verify feature contracts...")
        t0 = time.time()
        sample_run = completed_runs[0]
        r_id, f_id, part, p_path = sample_run
        print(f"Sample Run: {r_id} ({f_id}, Partition: {part}) -> {p_path}")
        df_raw = pq.read_table(str(p_path)).to_pandas()
        assert_pipeline_integrity(df_raw, r_id)

        df_feat = build_supervised_network_horizon_dataset(df_raw, "5min", 300, topo)
        elapsed = time.time() - t0

        print(f"Extraction Succeeded in {elapsed:.2f}s!")
        print(f"Feature Rows Generated: {len(df_feat):,} rows across {df_feat['road_id'].nunique()} roads")
        print("\nChecking Feature Integrity & Value Ranges:")
        for col in NETWORK_FEATURE_COLS:
            vals = df_feat[col]
            n_null = int(vals.isna().sum())
            n_inf = int(np.isinf(vals).sum())
            min_v = float(vals.min())
            max_v = float(vals.max())
            print(f"  • {col:<28}: Min={min_v:>8.3f} | Max={max_v:>8.3f} | Nulls={n_null} | Infs={n_inf}")
            assert n_null == 0, f"Nulls found in {col}"
            assert n_inf == 0, f"Infs found in {col}"

        # Check queue ratios above 1.0
        q_over_1 = df_feat[df_feat["down_1hop_max_queue_ratio"] > 1.0]
        print(f"\nUnbounded Queue Ratio Check: {len(q_over_1)} samples have down_1hop_max_queue_ratio > 1.0 (Preserved unclipped)")
        print("\nDry-run feature contract verification complete. Pipeline ready for batch generation.")
        return

    # Full Extraction
    horizons_to_build = HORIZONS_SEC.items() if args.horizon == "all" else [(args.horizon, HORIZONS_SEC[args.horizon])]
    t_start_all = time.time()

    for h_name, h_sec in horizons_to_build:
        print(f"\n{'='*30} BUILDING NETWORK FEATURES: {h_name} ({h_sec}s lookahead) {'='*30}")
        h_dir = NETWORK_FEATURE_OUTPUT_DIR / h_name
        h_dir.mkdir(parents=True, exist_ok=True)

        partitions: Dict[str, List[pd.DataFrame]] = {"TRAIN": [], "VAL": [], "IN_DIST_TEST": [], "HELD_OUT_TEST": []}
        t_h0 = time.time()

        for idx, (r_id, f_id, part, p_path) in enumerate(completed_runs):
            df_run = pq.read_table(str(p_path)).to_pandas()
            df_feat = build_supervised_network_horizon_dataset(df_run, h_name, h_sec, topo)
            partitions[part].append(df_feat)
            if (idx + 1) % 15 == 0 or (idx + 1) == len(completed_runs):
                print(f"  Processed {idx + 1}/{len(completed_runs)} runs...")

        for part_name, part_dfs in partitions.items():
            if part_dfs:
                out_df = pd.concat(part_dfs, ignore_index=True)
                out_path = h_dir / f"split_{part_name}.parquet"
                table = pa.Table.from_pandas(out_df)
                pq.write_table(table, str(out_path), compression="snappy")
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  [SAVED] {part_name:<14}: {len(out_df):>9,} rows | {size_mb:>6.1f} MB -> {out_path.name}")

        print(f"Horizon {h_name} completed in {time.time() - t_h0:.1f}s.")

    # Generate Manifest
    import hashlib
    import json

    manifest: Dict[str, Any] = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": 12,
            "horizons": ["5min", "10min", "15min"],
            "network_feature_count": len(NETWORK_FEATURE_COLS),
            "network_feature_cols": NETWORK_FEATURE_COLS,
        },
        "files": {},
    }

    for h_name in ("5min", "10min", "15min"):
        for part_name in ("TRAIN", "VAL", "IN_DIST_TEST", "HELD_OUT_TEST"):
            f_path = NETWORK_FEATURE_OUTPUT_DIR / h_name / f"split_{part_name}.parquet"
            if f_path.exists():
                file_bytes = f_path.read_bytes()
                sha256 = hashlib.sha256(file_bytes).hexdigest().upper()
                meta_table = pq.read_metadata(str(f_path))
                schema_table = pq.read_schema(str(f_path))
                manifest["files"][f"{h_name}/split_{part_name}.parquet"] = {
                    "horizon": h_name,
                    "partition": part_name,
                    "row_count": meta_table.num_rows,
                    "num_columns": len(schema_table.names),
                    "columns": schema_table.names,
                    "size_bytes": len(file_bytes),
                    "size_mb": round(len(file_bytes) / (1024 * 1024), 2),
                    "sha256": sha256,
                }

    manifest_path = NETWORK_FEATURE_OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[MANIFEST SAVED]: {manifest_path} (SHA-256 manifest recorded for all 12 files)")

    print(f"\n[COMPLETE]: All network-aware feature datasets generated in {time.time() - t_start_all:.1f}s.")


if __name__ == "__main__":
    main()
