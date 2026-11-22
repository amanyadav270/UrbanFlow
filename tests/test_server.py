"""Unit tests for UrbanFlow ASGI application endpoints without external httpx dependency."""
import asyncio
import json
import pytest
from starlette.datastructures import Headers
from urbanflow.server.app import index, api_status, api_topology, api_control


class DummyRequest:
    def __init__(self, json_body=None):
        self._json = json_body or {}
        self.headers = Headers({"content-type": "application/json"})

    async def json(self):
        return self._json


@pytest.mark.anyio
async def test_index_page():
    resp = await index(DummyRequest())
    assert resp.status_code == 200
    assert b"UrbanFlow AI" in resp.body
    assert b"viewport-container" in resp.body


@pytest.mark.anyio
async def test_api_status():
    resp = await api_status(DummyRequest())
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["status"] == "online"
    assert data["platform"] == "UrbanFlow AI"


@pytest.mark.anyio
async def test_api_topology():
    resp = await api_topology(DummyRequest())
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert "roads" in data
    assert "intersections" in data


def test_city_route_avoids_a_restricted_road_and_incident_expires():
    from urbanflow.core.kinematic_sim import KinematicSimulator
    from urbanflow.core.route_planner import plan_route

    sim = KinematicSimulator("metro")
    sim.set_demand((), "empty")
    source, destination = "INT_3_3", "INT_3_4"
    direct = plan_route(sim, source, destination, {})
    road, = direct["road_ids"]
    assert sim.inject_incident(road, severity=.95, duration_sec=90)
    events = {road: {"severity": .95, "end_time": 90}}
    detour = plan_route(sim, source, destination, events)
    assert road not in detour["road_ids"]
    assert sim._speed_limit(sim.roads[road], 0) == 2
    sim.sim_time = 90
    assert plan_route(sim, source, destination, events) == direct
    assert sim._speed_limit(sim.roads[road], 0) == sim.roads[road].speed_limit


@pytest.mark.anyio
async def test_api_control():
    req = DummyRequest(json_body={"action": "set_speed", "multiplier": 2.0})
    resp = await api_control(req)
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["status"] == "acknowledged"
    assert data["action"] == "set_speed"

    # Test map switching via control
    req_map = DummyRequest(json_body={"action": "load_map", "map_name": "procedural"})
    resp_map = await api_control(req_map)
    assert resp_map.status_code == 200
    assert json.loads(resp_map.body)["action"] == "load_map"

    # Test horizon switching via control
    req_horiz = DummyRequest(json_body={"action": "set_horizon", "horizon": "15min"})
    resp_horiz = await api_control(req_horiz)
    assert resp_horiz.status_code == 200
    assert json.loads(resp_horiz.body)["action"] == "set_horizon"

    # Reset back to bronx_fordham
    await api_control(DummyRequest(json_body={"action": "load_map", "map_name": "bronx_fordham"}))


@pytest.mark.anyio
async def test_study_uses_active_departures_and_rejects_stale_apply(monkeypatch):
    from importlib import import_module

    server = import_module("urbanflow.server.app")
    await server.api_control(DummyRequest({"action": "load_map", "map_name": "classroom_grid"}))
    await server.api_control(DummyRequest({"action": "load_scenario", "scenario": "surge"}))
    saved = await server.api_classroom_comparison(DummyRequest())
    assert json.loads(saved.body)["study"]["id"] == "preset_surge"
    assert json.loads(saved.body)["mode"] == "saved"
    server.simulator.set_demand(server.simulator.departures, "uploaded")
    selected = server.simulator.departures
    revision = server.broadcaster.demand_revision

    def compare(departures, scenario, demand_revision, on_progress):
        assert departures is selected
        assert scenario == "uploaded" and len(departures) == 457
        return {"demand_revision": demand_revision}

    monkeypatch.setattr(server, "compare_classroom", compare)
    result = await server.api_classroom_comparison(DummyRequest())
    assert json.loads(result.body)["demand_revision"] == revision
    assert server.simulator.departures is selected

    await server.api_control(DummyRequest({"action": "load_scenario", "scenario": "normal"}))
    result = await server.api_control(DummyRequest({
        "action": "set_controller", "controller": "ACTUATED", "demand_revision": revision,
    }))
    assert result.status_code == 409
