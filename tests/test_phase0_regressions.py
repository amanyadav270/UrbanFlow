import asyncio
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.server.websocket_feed import SimulationBroadcaster


def test_calibrated_prediction_uses_equivalent_named_features(monkeypatch):
    broadcaster = SimulationBroadcaster(KinematicSimulator("procedural"))
    model = broadcaster.controller.calibrator
    names = model.feature_names_in_
    row = np.arange(len(names), dtype=np.float32).reshape(1, -1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        expected = model.predict_proba(row)[0, 1]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="X does not have valid feature names")
        actual = model.predict_proba(pd.DataFrame(row, columns=names))[0, 1]
        monkeypatch.setattr(broadcaster.controller, "_booster", None)
        assert broadcaster.controller.predict_spillback_probability(row) == pytest.approx(expected)
    assert actual == pytest.approx(expected)


def test_first_routing_tick_logs_success_without_snapshot(monkeypatch, caplog):
    broadcaster = SimulationBroadcaster(KinematicSimulator("procedural"))

    def reroute(**kwargs):
        broadcaster.is_running = False
        return {
            "diverted_count": 1,
            "eligible_candidates_count": 1,
            "decisions": [{"diverted": True}],
        }

    monkeypatch.setattr(broadcaster.routing_controller, "evaluate_and_reroute", reroute)
    with caplog.at_level(logging.INFO):
        asyncio.run(asyncio.wait_for(broadcaster.run_loop(), timeout=5))
    assert broadcaster.cumulative_reroute_accepted == 1
    assert "[DYNAMIC_ROUTING_TICK]" in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
