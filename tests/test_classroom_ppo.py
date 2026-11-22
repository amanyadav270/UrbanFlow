import asyncio
import json

import pytest

from urbanflow.core.demand import classroom_demand
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.rl.env import SCHEMA, fingerprint, observe, requests
from urbanflow.rl.live import _load
from urbanflow.server.websocket_feed import SimulationBroadcaster


def test_features_and_topology_ignore_dynamic_phase():
    sim = KinematicSimulator("classroom_grid")
    original = fingerprint(sim)
    ids, local = observe(sim, False)
    _, neighbors = observe(sim, True)
    assert local.shape == neighbors.shape == (4, len(SCHEMA))
    assert not local[:, 11:].any()
    assert neighbors[:, 11:].any()
    sim.intersections[ids[0]].current_phase = 2
    assert fingerprint(sim) == original
    phase = sim.signal_controllers[ids[0]]
    sim.apply_signal_phases(requests(sim, ids, [1] * 4))
    assert phase.stage == "GREEN"
    assert phase.target_green_phase == 1 - phase.active_green_phase


def test_incompatible_weights_rejected_before_torch_import(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"variant": "local", "topology": "old"}))
    with pytest.raises(ValueError, match="incompatible"):
        _load(tmp_path, "new", "local", 1, 1)


def test_step_advances_exactly_one_decision_and_records_deferral(monkeypatch):
    async def run():
        sim = KinematicSimulator("classroom_grid")
        sim.set_demand(classroom_demand(sim, "normal"), "normal")
        broadcaster = SimulationBroadcaster(sim)
        broadcaster.is_paused = True
        node = next(iter(sim.signal_controllers))
        monkeypatch.setattr(broadcaster, "_compute_intelligence", lambda *a, **kw: {})
        monkeypatch.setattr(broadcaster.controller, "compute_phases", lambda **kwargs: {node: 1})
        task = asyncio.create_task(broadcaster.run_loop())
        try:
            for expected in (.05, 1.0):
                await broadcaster.handle_client_message({"action": "step_decision"})
                while broadcaster.step_pending:
                    await asyncio.sleep(.01)
                assert sim.sim_time == pytest.approx(expected)
                assert broadcaster.is_paused
            assert len(broadcaster.signal_decisions) == 2
            assert broadcaster.signal_decisions[-1]["outcome"] == "waiting for minimum green"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(asyncio.wait_for(run(), 5))
