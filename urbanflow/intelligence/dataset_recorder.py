"""Phase 2: Historical Traffic Observation Recorder & Reproducible Scenario Engine.

Stores time-series traffic state observations with scenario metadata, ensuring:
- Strict chronological ordering (no future leakage)
- Scenario reproducibility via deterministic seeds and demand configuration
- Explicit labelling of synthetic simulation data vs real sensor measurements
- Tabular feature extraction with chronological lag features (t-1, t-2, rolling statistics)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from urbanflow.intelligence.state_estimator import SegmentTrafficState


@dataclass
class ScenarioConfig:
    """Configuration for reproducible simulation scenarios."""
    scenario_id: str
    scenario_name: str
    random_seed: int
    duration_seconds: float
    base_spawn_interval: float          # Lower means higher baseline demand
    surge_multiplier: float = 1.0       # Demand surge multiplier
    surge_start_sec: float = -1.0
    surge_duration_sec: float = 0.0
    incident_road_id: Optional[str] = None
    incident_start_sec: float = -1.0
    incident_severity: float = 0.0      # 0.0 - 1.0 capacity reduction
    incident_duration_sec: float = 0.0
    description: str = ""
    data_source: str = "simulation_synthetic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetRecorder:
    """Records simulation state time series into SQLite/Parquet for model training."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            self.db_path = Path(__file__).resolve().parent.parent.parent / "data" / "datasets" / "traffic_history.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scenarios (
                    scenario_id TEXT PRIMARY KEY,
                    scenario_name TEXT,
                    random_seed INTEGER,
                    duration_sec REAL,
                    config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_id TEXT,
                    road_id TEXT,
                    timestamp REAL,
                    vehicle_count INTEGER,
                    density REAL,
                    avg_speed REAL,
                    flow REAL,
                    queue_length REAL,
                    queue_count INTEGER,
                    occupancy REAL,
                    travel_time REAL,
                    operational_capacity REAL,
                    capacity_utilization REAL,
                    signal_state TEXT,
                    incident_active INTEGER,
                    data_source TEXT,
                    FOREIGN KEY(scenario_id) REFERENCES scenarios(scenario_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_scenario_time ON observations(scenario_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_road_time ON observations(road_id, timestamp)")

    def record_scenario(self, config: ScenarioConfig):
        """Persists scenario metadata and reproducibility configuration."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scenarios 
                (scenario_id, scenario_name, random_seed, duration_sec, config_json, data_source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    config.scenario_id,
                    config.scenario_name,
                    config.random_seed,
                    config.duration_seconds,
                    json.dumps(config.to_dict()),
                    config.data_source,
                )
            )

    def record_observations_batch(self, scenario_id: str, states: List[SegmentTrafficState]):
        """Persists a batch of traffic state observations."""
        if not states:
            return

        rows = [
            (
                scenario_id,
                s.road_id,
                s.timestamp,
                s.vehicle_count,
                s.density_per_km_lane,
                s.average_speed_kmh,
                s.flow_veh_per_hour,
                s.queue_length_meters,
                s.queue_vehicle_count,
                s.occupancy_ratio,
                s.travel_time_sec,
                s.operational_capacity,
                s.capacity_utilization,
                s.signal_state,
                1 if s.incident_active else 0,
                s.data_source,
            )
            for s in states
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO observations (
                    scenario_id, road_id, timestamp, vehicle_count, density, avg_speed,
                    flow, queue_length, queue_count, occupancy, travel_time,
                    operational_capacity, capacity_utilization, signal_state, incident_active, data_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def load_scenario_dataframe(self, scenario_id: str) -> pd.DataFrame:
        """Loads observations into a pandas DataFrame sorted chronologically."""
        query = """
            SELECT scenario_id, road_id, timestamp, vehicle_count, density, avg_speed, flow,
                   queue_length, queue_count, occupancy, travel_time,
                   operational_capacity, capacity_utilization, signal_state, incident_active
            FROM observations
            WHERE scenario_id = ?
            ORDER BY timestamp ASC, road_id ASC
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(scenario_id,))
        return df

    @staticmethod
    def build_supervised_features(
        df: pd.DataFrame,
        horizon_sec: int,
        target_col: str = "avg_speed",
        lags: Tuple[int, ...] = (1, 2, 3),
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Builds chronological lag features for multi-step ahead prediction without future leakage.
        Vectorized with pandas groupby shifts for ultra-fast, leak-free feature construction.
        """
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        dfs = []
        group_cols = ["scenario_id", "road_id"] if "scenario_id" in df.columns else ["road_id"]

        for _, group in df.groupby(group_cols):
            group = group.sort_values("timestamp").copy()
            if len(group) < 15:
                continue

            diffs = group["timestamp"].diff().dropna()
            pos_diffs = diffs[diffs > 0]
            dt_step = float(pos_diffs.median()) if not pos_diffs.empty else 5.0
            steps_ahead = max(1, int(round(horizon_sec / max(0.1, dt_step))))

            group["target_future"] = group[target_col].shift(-steps_ahead)

            for lag in lags:
                group[f"{target_col}_lag_{lag}"] = group[target_col].shift(lag)
                group[f"density_lag_{lag}"] = group["density"].shift(lag)
                group[f"queue_lag_{lag}"] = group["queue_length"].shift(lag)

            group[f"{target_col}_roll_mean"] = group[target_col].rolling(4, min_periods=1).mean()
            group[f"{target_col}_roll_std"] = group[target_col].rolling(4, min_periods=1).std().fillna(0.0)

            group["signal_is_red"] = (group["signal_state"] == "r").astype(int)
            group["incident_active"] = group["incident_active"].astype(int)
            group["current_val"] = group[target_col]

            valid = group.dropna(subset=["target_future", f"{target_col}_lag_{max(lags)}"]).copy()
            if not valid.empty:
                dfs.append(valid)

        if not dfs:
            return pd.DataFrame(), pd.Series(dtype=float)

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        raw_feature_cols = [
            "current_val", "density", "flow", "queue_length", "occupancy",
            "capacity_utilization", "signal_is_red", "incident_active",
            f"{target_col}_roll_mean", f"{target_col}_roll_std"
        ] + [f"{target_col}_lag_{l}" for l in lags] + [f"density_lag_{l}" for l in lags] + [f"queue_lag_{l}" for l in lags]

        # Deduplicate column names while preserving order
        feature_cols = list(dict.fromkeys(raw_feature_cols))

        X = merged[feature_cols]
        y = merged["target_future"]
        return X, y
