import asyncio
import json
import time

import pytest

from urbanflow.core.demand import classroom_demand
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.rl_benchmarks import read_benchmarks
from urbanflow.server import app as server
from urbanflow.server.websocket_feed import SimulationBroadcaster


@pytest.mark.anyio
async def test_preset_uses_matched_replay(monkeypatch):
    sim = KinematicSimulator("classroom_grid")
    demand = classroom_demand(sim, "normal", seed=42)
    sim.set_demand(demand, "normal")
    monkeypatch.setattr(server, "simulator", sim)

    def compare(departures, scenario, revision, on_progress, *, include_ppo):
        assert include_ppo
        assert departures == demand
        assert scenario == "normal"
        return {"kind": "controller", "scheduled": len(departures)}

    monkeypatch.setattr(server, "compare_classroom", compare)
    matched = await server.api_matched_comparison(None)
    archived = await server.api_classroom_comparison(None)
    assert json.loads(matched.body) == {"kind": "controller", "scheduled": len(demand)}
    assert json.loads(archived.body)["mode"] == "saved"


def test_archive_exposes_paired_seeds():
    study, = read_benchmarks(("grid_test_sym",))["studies"]
    assert study["engine"] == "SUMO"
    assert len(study["seeds"]) == study["seed_count"]


@pytest.mark.anyio
async def test_routing_comparison_accepts_loaded_preset(monkeypatch):
    sim = KinematicSimulator("classroom_grid")
    demand = classroom_demand(sim, "normal")
    sim.set_demand(demand, "normal")
    monkeypatch.setattr(server, "simulator", sim)

    def compare(departures, scenario, revision, on_progress, controller):
        assert departures == demand
        assert scenario == "normal"
        return {"kind": "routing"}

    monkeypatch.setattr(server, "compare_routing", compare)
    response = await server.api_routing_comparison(None)
    assert response.status_code == 200
    assert json.loads(response.body) == {"kind": "routing"}


@pytest.mark.anyio
async def test_closing_test_drive_stream_cancels_replay(monkeypatch):
    def run(*args, on_reference, cancelled, on_samples):
        on_samples({"trace": [[0, 0, 0, 0]], "traffic": []})
        deadline = time.monotonic() + 2
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(.001)
        assert cancelled()

    monkeypatch.setattr(server, "run_route_experiment", run)
    response = await server.stream_route_experiment((), "test-cancel")
    stream = response.body_iterator
    assert json.loads(await anext(stream))["type"] == "samples"
    await stream.aclose()
    async def wait_for_cleanup():
        while "test-cancel" in server.active_route_streams:
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait_for_cleanup(), timeout=1)


@pytest.mark.anyio
async def test_routing_demo_preserves_map_demand_and_controller_without_detour():
    sim = KinematicSimulator("procedural")
    demand = classroom_demand(sim, "normal")
    sim.set_demand(demand, "normal")
    broadcaster = SimulationBroadcaster(sim)
    broadcaster.set_controller("ACTUATED")
    demo = await broadcaster.handle_client_message({"action": "start_routing_demo"})
    assert demo["map_name"] == sim.map_name == "procedural"
    assert sim.departures == demand
    assert broadcaster.active_controller_name == "ACTUATED"
    assert demo["road"] in broadcaster.active_incidents
    assert not broadcaster.is_paused
