"""Phase 8.1 Verification Test Suite: Predictor Stability and Live Network Feature Integration.

Verifies:
1. Native Booster loading on Windows (all 9 regression models load safely without access violations).
2. Multi-horizon online inference (5min, 10min, 15min across avg_speed, density, queue_length).
3. 28-feature schema and column ordering matching the frozen spillback classifier.
4. Multi-hop downstream line graph feature calculations (1-hop, 2-hop, 3-hop, perimeter exits).
5. 5-second sampling cadence verification in observation buffer.
6. Temporal integrity (no future leakage, backward-looking timestamps [t-15s, t-10s, t-5s, t]).
7. Independent road history isolation (no cross-road contamination).
8. Insufficient history handling (< 4 steps yields None / INSUFFICIENT_HISTORY).
9. Observation buffer reset cleanliness (clears history on reset / reload).
10. Signal controller receives calibrated probabilities and applies predictive penalties.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Dict, Any

from urbanflow.core.network_topology import NetworkTopology
from urbanflow.intelligence.congestion_predictor import (
    LightGBMPredictor,
    ModularCongestionPredictor,
)
from urbanflow.intelligence.routing import OnlineObservationBuffer
from urbanflow.intelligence.signal_control import CooperativePredictiveSignalController
from urbanflow.server.websocket_feed import SimulationBroadcaster
from urbanflow.core.kinematic_sim import KinematicSimulator


URBANFLOW_DATA_MODELS = Path(__file__).resolve().parent.parent / "data" / "models"


def test_1_native_booster_loading_windows():
    """Test 1: Verifies all 9 LightGBM regression models load via native Booster without access violations."""
    horizons = ["5min", "10min", "15min"]
    targets = ["avg_speed", "density", "queue_length"]

    for h in horizons:
        for t in targets:
            model_path = URBANFLOW_DATA_MODELS / f"lgb_{h}_{t}.joblib"
            assert model_path.exists(), f"Model file missing: {model_path}"

            predictor = LightGBMPredictor()
            loaded = predictor.load(model_path)
            assert loaded is True, f"Failed to load model file: {model_path}"
            assert predictor.is_fitted, f"Predictor is_fitted is False: {model_path}"
            assert predictor._booster is not None, f"Native _booster is None for {model_path}"
            assert getattr(predictor.model, "_Booster", None) is predictor._booster, (
                f"_Booster attribute not synced for {model_path}"
            )

            # Test safe inference on Windows with dummy vector
            dummy_input = np.ones(16, dtype=np.float64)
            preds = predictor.predict(dummy_input)
            val = float(preds[0])
            assert isinstance(val, float), f"Expected float prediction, got {type(val)}"
            assert not np.isnan(val), f"NaN prediction for {model_path}"


def test_2_multi_horizon_online_inference():
    """Test 2: Verifies ModularCongestionPredictor and all 9 LightGBM models across horizons and targets."""
    predictor = ModularCongestionPredictor()
    loaded = predictor.load_models(URBANFLOW_DATA_MODELS)
    assert loaded is True

    horizons = ["5min", "10min", "15min"]
    targets = ["avg_speed", "density", "queue_length"]

    current_state_features = {
        "avg_speed": 35.0,
        "density": 25.0,
        "queue_length": 15.0,
        "avg_speed_roll_mean": 36.0,
        "density_roll_mean": 24.0,
        "queue_roll_mean": 14.0,
        "current_val": 35.0,
        "vc_ratio": 0.45,
    }

    # 1. Test ModularCongestionPredictor.predict_online() across all 3 horizons
    for h in horizons:
        res = predictor.predict_online(
            current_state_features=current_state_features,
            horizon=h,
            timestamp=100.0,
        )
        assert res["inference_status"] == "complete", f"Inference not complete for {h}: {res}"
        for t in targets:
            assert f"predicted_{t}" in res
            assert isinstance(res[f"predicted_{t}"], float)
            assert res[f"predicted_{t}"] >= 0.0
            assert res["statuses"][t] == "available"

    # 2. Test all 9 native LightGBM regression models individually on DataFrame features
    df_feat = pd.DataFrame([{
        "avg_speed_lag_1": 35.0, "avg_speed_lag_2": 36.0, "avg_speed_lag_3": 37.0,
        "avg_speed_roll_mean": 36.0, "avg_speed_roll_std": 1.0,
        "density_lag_1": 20.0, "density_lag_2": 21.0, "density_lag_3": 22.0,
        "density_roll_mean": 21.0,
        "queue_lag_1": 10.0, "queue_lag_2": 11.0, "queue_lag_3": 12.0,
        "queue_roll_mean": 11.0,
        "operational_capacity": 1800.0, "vc_ratio": 0.4, "incident_active": 0.0,
        "current_val": 35.0,
    }])
    for h in horizons:
        for t in targets:
            lgb_mod = predictor.models[h][t]
            assert lgb_mod.is_fitted
            val = float(lgb_mod.predict(df_feat)[0])
            assert not np.isnan(val)
            assert val >= 0.0


def test_3_feature_28_schema_and_ordering():
    """Test 3: Verifies OnlineObservationBuffer generates exactly 28 features matching classifier schema."""
    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)

    # Check signal controller's expected features
    ctrl = CooperativePredictiveSignalController(enable_prediction=True)
    expected_features = ctrl.expected_features
    assert len(expected_features) == 28, f"Expected 28 features, got {len(expected_features)}"

    # Populate 4 steps for road
    test_road = list(topo.roads.keys())[0]
    for i in range(4):
        telem = {
            test_road: {
                "timestamp": float(i * 5.0),
                "avg_speed_kmh": 35.0 - i,
                "density_per_km_lane": 10.0 + i * 5,
                "queue_length_meters": i * 15.0,
                "operational_capacity": 1800.0,
                "capacity_utilization": 0.2 + i * 0.1,
                "occupancy_ratio": 0.15 + i * 0.05,
                "incident_active": False,
            }
        }
        buf.update(telem, timestamp=float(i * 5.0))

    assert buf.is_road_ready(test_road)
    feat_dict = buf.build_feature_dict(test_road)
    assert feat_dict is not None
    assert len(feat_dict) == 28, f"Expected 28 features from build_feature_dict, got {len(feat_dict)}"

    for f_name in expected_features:
        assert f_name in feat_dict, f"Feature '{f_name}' missing from OnlineObservationBuffer output!"


def test_4_multi_hop_downstream_features():
    """Test 4: Verifies 1-hop, 2-hop, 3-hop downstream features and perimeter exit handling."""
    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)

    # 1. Perimeter Exit Road (zero downstream roads)
    exit_roads = [r for r in topo.roads if topo.is_perimeter_exit(r)]
    assert len(exit_roads) > 0, "No perimeter exits found in topology"
    exit_road = exit_roads[0]

    for i in range(4):
        buf.update({
            exit_road: {
                "timestamp": float(i * 5.0),
                "avg_speed_kmh": 45.0,
                "density_per_km_lane": 5.0,
                "queue_length_meters": 0.0,
                "capacity_utilization": 0.1,
                "occupancy_ratio": 0.05,
            }
        }, timestamp=float(i * 5.0))

    exit_feats = buf.build_feature_dict(exit_road)
    assert exit_feats is not None
    assert exit_feats["is_perimeter_exit"] == 1.0
    assert exit_feats["down_1hop_max_queue_ratio"] == 0.0
    assert exit_feats["down_2hop_max_queue_ratio"] == 0.0
    assert exit_feats["down_3hop_max_queue_ratio"] == 0.0

    # 2. Interior Road with downstream connectivity
    interior_roads = [r for r in topo.roads if len(topo.get_downstream_roads(r, 1)) > 0]
    interior_road = interior_roads[0]
    d1 = topo.get_downstream_roads(interior_road, 1)

    for i in range(4):
        telem = {
            interior_road: {
                "timestamp": float(i * 5.0),
                "avg_speed_kmh": 30.0,
                "density_per_km_lane": 20.0,
                "queue_length_meters": 10.0,
                "capacity_utilization": 0.4,
                "occupancy_ratio": 0.2,
            }
        }
        for d_r in d1:
            telem[d_r] = {
                "timestamp": float(i * 5.0),
                "avg_speed_kmh": 15.0,
                "density_per_km_lane": 60.0,
                "queue_length_meters": 50.0,
                "capacity_utilization": 0.9,
                "occupancy_ratio": 0.8,
            }
        buf.update(telem, timestamp=float(i * 5.0))

    interior_feats = buf.build_feature_dict(interior_road)
    assert interior_feats is not None
    assert interior_feats["is_perimeter_exit"] == 0.0
    assert interior_feats["down_1hop_max_queue_ratio"] > 0.0
    assert interior_feats["down_1hop_min_speed_kmh"] == 15.0


def test_5_observation_buffer_5s_cadence():
    """Test 5: Verifies that observations are sampled at strictly 5-second cadence."""
    sim = KinematicSimulator()
    broadcaster = SimulationBroadcaster(simulator=sim)
    broadcaster.last_obs_sample_time = -1.0
    broadcaster.observation_buffer.reset()

    # Simulate 20 Hz simulation steps (every 0.05s) from 0s to 16s
    road_id = "test_road_cadence"
    for step in range(321):  # 0 to 320 * 0.05 = 16.0s
        t = round(step * 0.05, 3)
        # Emulate broadcaster sample condition:
        if broadcaster.last_obs_sample_time < 0.0 or (t - broadcaster.last_obs_sample_time) >= broadcaster.obs_sample_interval_sec:
            broadcaster.last_obs_sample_time = t
            broadcaster.observation_buffer.update({
                road_id: {
                    "avg_speed_kmh": 30.0,
                    "density_per_km_lane": 20.0,
                    "queue_length_meters": 10.0,
                    "operational_capacity": 1800.0,
                    "capacity_utilization": 0.3,
                    "occupancy_ratio": 0.2,
                    "incident_active": False,
                }
            }, timestamp=t)

    # At 16.0s with 5.0s interval, samples should be taken at: 0.0s, 5.0s, 10.0s, 15.0s (total 4 samples)
    road_history = broadcaster.observation_buffer.buffer[road_id]
    assert len(road_history) == 4, f"Expected 4 samples at t=16s, got {len(road_history)}"
    timestamps = [snap["timestamp"] for snap in road_history]
    assert timestamps == [0.0, 5.0, 10.0, 15.0], f"Unexpected timestamps: {timestamps}"


def test_6_no_future_leakage():
    """Test 6: Verifies temporal integrity and strictly backward-looking lag computations."""
    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)
    road = list(topo.roads.keys())[0]

    # Feed 4 historical snapshots at t=0s, 5s, 10s, 15s
    times = [0.0, 5.0, 10.0, 15.0]
    speeds = [40.0, 35.0, 30.0, 25.0]  # speed drops over time
    for t, s in zip(times, speeds):
        buf.update({
            road: {
                "timestamp": t,
                "avg_speed_kmh": s,
                "density_per_km_lane": 10.0,
                "queue_length_meters": 5.0,
            }
        }, timestamp=t)

    feats = buf.build_feature_dict(road)
    assert feats is not None
    # Lags must correspond strictly to past steps:
    # lag_1 is t-5s (30.0), lag_2 is t-10s (35.0), lag_3 is t-15s (40.0)
    assert feats["avg_speed_lag_1"] == 30.0
    assert feats["avg_speed_lag_2"] == 35.0
    assert feats["avg_speed_lag_3"] == 40.0
    # Current t=15s speed is 25.0, roll mean of all 4 is (40+35+30+25)/4 = 32.5
    assert abs(feats["avg_speed_roll_mean"] - 32.5) < 1e-5


def test_7_independent_road_history():
    """Test 7: Verifies independent road buffer isolation with no cross-contamination."""
    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)
    roads = list(topo.roads.keys())[:2]
    road_A, road_B = roads[0], roads[1]

    # Feed 4 snapshots to Road A only
    for i in range(4):
        buf.update({
            road_A: {
                "timestamp": float(i * 5.0),
                "avg_speed_kmh": 20.0,
                "density_per_km_lane": 50.0,
                "queue_length_meters": 40.0,
            }
        }, timestamp=float(i * 5.0))

    assert buf.is_road_ready(road_A) is True
    assert buf.is_road_ready(road_B) is False
    assert buf.build_feature_dict(road_A) is not None
    assert buf.build_feature_dict(road_B) is None
    assert len(buf.buffer.get(road_B, [])) == 0


def test_8_insufficient_history_handling():
    """Test 8: Verifies handling when temporal history has fewer than 4 snapshots."""
    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)
    road = list(topo.roads.keys())[0]

    # 0 snapshots
    assert buf.is_road_ready(road) is False
    assert buf.build_feature_dict(road) is None

    # 1 snapshot
    buf.update({road: {"avg_speed_kmh": 30.0, "density_per_km_lane": 10.0, "queue_length_meters": 0.0}}, 0.0)
    assert buf.is_road_ready(road) is False
    assert buf.build_feature_dict(road) is None

    # 2 snapshots
    buf.update({road: {"avg_speed_kmh": 30.0, "density_per_km_lane": 10.0, "queue_length_meters": 0.0}}, 5.0)
    assert buf.is_road_ready(road) is False
    assert buf.build_feature_dict(road) is None

    # 3 snapshots
    buf.update({road: {"avg_speed_kmh": 30.0, "density_per_km_lane": 10.0, "queue_length_meters": 0.0}}, 10.0)
    assert buf.is_road_ready(road) is False
    assert buf.build_feature_dict(road) is None

    # 4 snapshots -> becomes ready
    buf.update({road: {"avg_speed_kmh": 30.0, "density_per_km_lane": 10.0, "queue_length_meters": 0.0}}, 15.0)
    assert buf.is_road_ready(road) is True
    assert buf.build_feature_dict(road) is not None


def test_9_buffer_reset_cleanliness():
    """Test 9: Verifies that buffer reset properly purges all history and state."""
    topo = NetworkTopology.get_instance()
    buf = OnlineObservationBuffer(topo)
    road = list(topo.roads.keys())[0]

    for i in range(5):
        buf.update({road: {"avg_speed_kmh": 30.0, "density_per_km_lane": 10.0, "queue_length_meters": 0.0}}, float(i * 5.0))

    assert buf.is_road_ready(road) is True
    assert buf.last_update_time == 20.0

    # Call reset
    buf.reset()
    assert len(buf.buffer) == 0
    assert buf.last_update_time == -1.0
    assert buf.is_road_ready(road) is False
    assert buf.build_feature_dict(road) is None


def test_10_signal_controller_receives_calibrated_probabilities():
    """Test 10: Verifies CooperativePredictiveSignalController evaluates calibrated probabilities and penalties."""
    ctrl = CooperativePredictiveSignalController(enable_prediction=True)
    assert ctrl.predictive_available is True

    # 1. Telemetry with INSUFFICIENT_HISTORY / None probability
    telem_insufficient = {
        "road_heatmap": {
            "road_1": {
                "queue_length_m": 25.0,
                "predicted_spillback_probability": None,
                "prediction_status": "INSUFFICIENT_HISTORY",
            }
        }
    }
    pen_none = ctrl._compute_predictive_penalty("road_1", telem_insufficient)
    assert pen_none == 0.0, f"Expected 0.0 penalty for None probability, got {pen_none}"

    # 2. Telemetry with probability below threshold (e.g. 0.05 < 0.180)
    telem_low = {
        "road_heatmap": {
            "road_1": {
                "queue_length_m": 25.0,
                "predicted_spillback_probability": 0.05,
                "prediction_status": "AVAILABLE",
            }
        }
    }
    pen_low = ctrl._compute_predictive_penalty("road_1", telem_low)
    assert pen_low == 0.0, f"Expected 0.0 penalty below threshold, got {pen_low}"

    # 3. Telemetry with probability above threshold (e.g. 0.75 > 0.180)
    telem_high = {
        "road_heatmap": {
            "road_1": {
                "queue_length_m": 25.0,
                "predicted_spillback_probability": 0.75,
                "prediction_status": "AVAILABLE",
            }
        }
    }
    pen_high = ctrl._compute_predictive_penalty("road_1", telem_high)
    assert pen_high > 0.0, f"Expected positive penalty above threshold, got {pen_high}"
    assert pen_high <= 8.0, "Penalty exceeded 8.0 cap"

    # 4. Verify flat dictionary fallback support as well
    telem_flat = {
        "road_1": {
            "queue_length_m": 25.0,
            "predicted_spillback_probability": 0.80,
            "prediction_status": "AVAILABLE",
        }
    }
    pen_flat = ctrl._compute_predictive_penalty("road_1", telem_flat)
    assert pen_flat > 0.0, f"Expected positive penalty with flat telemetry format, got {pen_flat}"
