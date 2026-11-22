"""ASGI Application Server for UrbanFlow AI."""
from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
from pathlib import Path
from threading import Event
from starlette.applications import Starlette
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from urbanflow.config import WEB_DIR
from urbanflow.core.kinematic_sim import KinematicSimulator
from urbanflow.core.demand import STRESS_SCENARIOS, classroom_demand, parse_demand, write_demand_csv
from urbanflow.core.observed_demand import COUNT_SOURCE, COUNTS, OBSERVED_WINDOWS
from urbanflow.core.rl_benchmarks import ARCHIVE, PRESET_STUDIES, SCENARIOS, read_benchmarks
from urbanflow.core.route_planner import plan_route
from urbanflow.core.route_experiment import run_route_experiment, validate_route
from urbanflow.core.classroom_evaluation import compare_classroom, compare_routing
from urbanflow.rl.live import POLICIES, load as load_policy
from urbanflow.server.websocket_feed import (
    SimulationBroadcaster,
    CONTROLLER_REGIMES,
    ROUTING_MODES,
)

# Global simulation instance
simulator = KinematicSimulator("classroom_grid")
simulator.set_demand(classroom_demand(simulator, "rush_hour"), "rush_hour")
broadcaster = SimulationBroadcaster(simulator)
broadcaster.is_paused = True
broadcaster.set_controller("FIXED_TIME")
broadcaster.prepare_classroom_model()
comparison_lock = asyncio.Lock()
comparison_progress = {}


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Manages background simulation loop lifecycle."""
    broadcaster.start()
    yield
    broadcaster.stop()


async def index(request):
    """Serves the primary 3D Operations Command Center HTML page."""
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>UrbanFlow AI — Web UI Initializing</h1>", status_code=200)


async def api_status(request):
    """Health check and simulator status endpoint."""
    return JSONResponse({
        "status": "online",
        "platform": "UrbanFlow AI",
        "simulator": "KinematicSimulator",
        "map_name": getattr(simulator, "map_name", "classroom_grid"),
        "active_clients": len(broadcaster.clients),
        "paused": broadcaster.is_paused,
        "speed": broadcaster.sim_speed_multiplier,
        "demand": simulator.demand_counts(),
        "active_controller": broadcaster.active_controller_name,
        "controller_name": broadcaster.controller_telemetry["controller_name"],
        "controller_error": broadcaster.control_error,
        "controlled_intersections": broadcaster.controller_telemetry["controlled_intersections_count"],
        "controller_status": broadcaster.controller_telemetry.get("status", "OPERATIONAL"),
        "predictive_status": broadcaster.controller_telemetry.get("predictive_status", "N/A"),
        "available_controllers": list(CONTROLLER_REGIMES.keys()),
        "active_routing_mode": broadcaster.active_routing_mode,
        "routing_status": broadcaster.routing_telemetry.get("status", "OFF"),
        "available_routing_modes": list(ROUTING_MODES.keys()),
        "reroutes_accepted": broadcaster.cumulative_reroute_accepted,
        "reroutes_rejected": broadcaster.cumulative_reroute_rejected,
    })


async def api_routing(request):
    """Returns available routing modes, active mode, and detailed routing telemetry."""
    return JSONResponse({
        "active_mode": broadcaster.active_routing_mode,
        "routing_telemetry": broadcaster.routing_telemetry,
        "available_modes": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "is_active": (k == broadcaster.active_routing_mode),
            }
            for k, v in ROUTING_MODES.items()
        ],
    })


async def api_set_routing(request):
    """Sets the active dynamic routing mode ('OFF', 'REACTIVE', 'PREDICTIVE')."""
    try:
        body = await request.json()
        mode = str(body.get("mode", "")).strip().upper()
        if mode not in ROUTING_MODES:
            return JSONResponse({
                "status": "error",
                "message": f"Invalid routing mode '{mode}'. Must be one of: {list(ROUTING_MODES.keys())}",
            }, status_code=400)
        telemetry = broadcaster.set_routing_mode(mode)
        return JSONResponse({
            "status": "acknowledged",
            "action": "set_routing_mode",
            "active_mode": broadcaster.active_routing_mode,
            "routing_info": telemetry,
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


async def api_controllers(request):
    """Returns available signal controller regimes and currently active controller status."""
    return JSONResponse({
        "active_controller": broadcaster.active_controller_name,
        "controller_telemetry": broadcaster.controller_telemetry,
        "available_controllers": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "is_active": (k == broadcaster.active_controller_name),
            }
            for k, v in CONTROLLER_REGIMES.items()
        ],
    })


AVAILABLE_SCENARIOS = [
    {"id": "normal", "name": "Normal", "description": "Lower-volume balanced arrivals"},
    {"id": "rush_hour", "name": "Rush Hour", "description": "Demand increases and favours east-west approaches"},
    {"id": "surge", "name": "Peak Demand", "description": "High arrival rate"},
    {"id": "lane_closure", "name": "Lane Closure", "description": "Temporary closure of one internal lane"},
    {"id": "incident", "name": "Road Incident", "description": "Temporary speed restriction on an internal road"},
    {"id": "gridlock", "name": "Downstream Gridlock", "description": "Temporary downstream exit blockage"},
    *({"id": key, "name": name, "source": "synthetic"} for key, (name, _) in STRESS_SCENARIOS.items()),
    *({"id": key, "name": name, "source": COUNT_SOURCE, "window": f"2013-01-{day:02} {hour:02}:00"}
      for key, (day, hour, name) in OBSERVED_WINDOWS.items()),
]


async def api_scenarios(request):
    """List classroom demand schedules."""
    return JSONResponse({"status": "online", "available_scenarios": AVAILABLE_SCENARIOS})


async def api_observed_counts(request):
    return FileResponse(COUNTS, media_type="text/csv", filename=COUNTS.name)


async def api_export_demand(request):
    if not simulator.departures:
        return JSONResponse({"message": "Load a demand schedule to export it."}, status_code=400)
    filename = f"urbanflow-{simulator.map_name}-demand.csv"
    return Response(write_demand_csv(simulator.departures), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


async def api_demand(request):
    map_revision = broadcaster.demand_revision
    try:
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 2_000_000:
                return JSONResponse({"message": "Demand files are limited to 2 MB."}, status_code=413)
        if broadcaster.demand_revision != map_revision:
            raise ValueError("The selected map or demand changed during upload. Retry on the intended network.")
        departures = parse_demand(simulator, data.decode("utf-8"), request.query_params.get("format"))
        simulator.set_demand(departures, "uploaded")
        broadcaster.demand_revision += 1
        broadcaster.reset_run()
        broadcaster.is_paused = True
        return JSONResponse({**simulator.demand_counts(), "demand_revision": broadcaster.demand_revision})
    except ValueError as exc:
        return JSONResponse({"message": str(exc)}, status_code=400)


def set_comparison_progress(progress):
    global comparison_progress
    comparison_progress = progress


async def api_classroom_comparison_progress(request):
    return JSONResponse(comparison_progress)


async def api_classroom_comparison(request):
    return await run_comparison(routing=False)


async def api_matched_comparison(request):
    return await run_comparison(routing=False, live=True)


async def api_learned_controllers(request):
    choices = {}
    for key in POLICIES:
        try:
            load_policy(simulator, key)
        except ValueError as error:
            choices[key] = str(error)
        else:
            choices[key] = None
    return JSONResponse({"map_name": simulator.map_name, "choices": choices,
                         "experimental": simulator.map_name != "classroom_grid"})


async def api_held_out(request):
    directory = ARCHIVE.parent / "classroom_ppo" / "evaluation"
    download = request.query_params.get("format") == "csv"
    result = directory / ("per-seed.csv" if download else "results.json")
    if not result.is_file():
        return JSONResponse({"message": "Held-out evaluation is not available."}, status_code=404)
    return FileResponse(result, filename=result.name if download else None)


async def api_routing_comparison(request):
    return await run_comparison(routing=True)


async def run_comparison(*, routing, live=False):
    if not simulator.departures:
        return JSONResponse({"message": "Load a demand schedule before starting a comparison."}, status_code=400)
    if simulator.demand_name != "uploaded" and not live and not routing:
        key = PRESET_STUDIES.get(simulator.demand_name)
        if not key:
            return JSONResponse({"message": "Upload a demand CSV or JSON to run this live comparison."}, status_code=400)
        study, = read_benchmarks((key,))["studies"]
        return JSONResponse({"mode": "saved", "study": study, "demand_revision": broadcaster.demand_revision})
    if comparison_lock.locked():
        return JSONResponse({"message": "A comparison is already running."}, status_code=409)
    async with comparison_lock:
        demand, scenario, revision = simulator.departures, simulator.demand_name, broadcaster.demand_revision
        selected_controller = broadcaster.active_controller_name
        map_options = {"map_name": simulator.map_name} if simulator.map_name != "classroom_grid" else {}

        def notify(progress):
            set_comparison_progress({**progress, "kind": "routing" if routing else "controller"})

        try:
            if routing:
                report = await run_in_threadpool(compare_routing, demand, scenario, revision, notify, selected_controller, **map_options)
            else:
                options = {"include_ppo": True} if live else {}
                report = await run_in_threadpool(compare_classroom, demand, scenario, revision, notify, **map_options, **options)
            return JSONResponse(report)
        finally:
            set_comparison_progress({})


async def api_rl_benchmarks(request):
    return JSONResponse(read_benchmarks())


async def api_rl_benchmark_csv(request):
    key = request.path_params["scenario"]
    if key not in SCENARIOS:
        return JSONResponse({"message": "Benchmark scenario not found."}, status_code=404)
    filename = f"{key}.csv"
    return FileResponse(ARCHIVE / filename, media_type="text/csv", filename=filename)


async def api_topology(request):
    """Returns network road and intersection definitions."""
    return JSONResponse(simulator.get_network_topology())


async def api_plan_route(request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or payload.get("map_name") != simulator.map_name:
            raise ValueError("The network changed. Choose the route again.")
        return JSONResponse(plan_route(simulator, payload.get("from"), payload.get("to"), broadcaster.active_incidents))
    except ValueError as error:
        return JSONResponse({"message": str(error)}, status_code=400)


async def api_route_experiment(request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or payload.get("map_name") != simulator.map_name:
            raise ValueError("The network changed. Find the route again.")
        roads = payload.get("road_ids")
        validate_route(simulator, roads, payload.get("from"), payload.get("to"))
        if request.query_params.get("stream") == "1":
            args = route_experiment_args(roads)
            session = payload.get("session")
            if not isinstance(session, str) or len(session) > 64:
                session = None
            return await stream_route_experiment(args, session)
        if comparison_lock.locked():
            return JSONResponse({"message": "Another experiment is running. Try again shortly."}, status_code=409)
        async with comparison_lock:
            return JSONResponse(await run_in_threadpool(run_route_experiment, *route_experiment_args(roads)))
    except ValueError as error:
        return JSONResponse({"message": str(error)}, status_code=400)


def route_experiment_args(roads):
    return (simulator.map_name, simulator.demand_name, tuple(simulator.departures or ()),
            roads, broadcaster.active_controller_name,
            {road: {"severity": event["severity"], "end_time": event["end_time"] - simulator.sim_time}
             for road, event in broadcaster.active_incidents.items()
             if event["end_time"] > simulator.sim_time})


active_route_streams = {}


async def stream_route_experiment(args, session=None):
    loop = asyncio.get_running_loop()
    messages = asyncio.Queue()
    stop = Event()
    if session:
        previous = active_route_streams.get(session)
        if previous:
            previous.set()
        active_route_streams[session] = stop

    def publish_reference(report):
        end = {key: value for key, value in report.items() if key not in ("trace", "traffic")}
        loop.call_soon_threadsafe(messages.put_nowait, {"type": "replay", "report": end})

    def publish_samples(samples):
        loop.call_soon_threadsafe(messages.put_nowait, {"type": "samples", "report": samples})

    async def execute():
        try:
            report = await run_in_threadpool(run_route_experiment, *args, on_reference=publish_reference,
                                             cancelled=stop.is_set, on_samples=publish_samples)
            if report is None:
                return
            report.pop("trace")
            report.pop("traffic")
            messages.put_nowait({"type": "comparison", "report": report})
        except ValueError as error:
            messages.put_nowait({"type": "error", "message": str(error)})
        except Exception:
            messages.put_nowait({"type": "error", "message": "The route experiment failed. Please retry."})
            raise
        finally:
            if session and active_route_streams.get(session) is stop:
                active_route_streams.pop(session)
            messages.put_nowait(None)

    task = asyncio.create_task(execute())

    async def generate():
        try:
            while (message := await messages.get()) is not None:
                yield json.dumps(message, separators=(",", ":")) + "\n"
            await task
        finally:
            stop.set()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def api_control(request):
    """Receives REST-based control actions."""
    try:
        body = await request.json()
        action = body.get("action")
        if action == "set_controller":
            revision = body.get("demand_revision")
            if revision is not None and revision != broadcaster.demand_revision:
                return JSONResponse({"message": "The demand changed. Run the comparison again before applying it."},
                                    status_code=409)
            ctrl = str(body.get("controller", "")).strip().upper()
            if ctrl not in CONTROLLER_REGIMES and ctrl not in POLICIES:
                return JSONResponse({
                    "status": "error",
                    "message": f"Invalid controller '{ctrl}'. Must be one of: {list(CONTROLLER_REGIMES.keys())}",
                }, status_code=400)
            if ctrl == "PPO_NEIGHBOR" and simulator.map_name == "classroom_grid":
                broadcaster.activate_classroom_model()
            else:
                broadcaster.set_controller(ctrl)
                broadcaster._update_controller_telemetry()
            if ctrl in POLICIES and not (ctrl == "PPO_NEIGHBOR" and simulator.map_name == "classroom_grid"):
                simulator.reset()
                broadcaster.demand_revision += 1
                broadcaster.reset_run()
                broadcaster.is_paused = True
            return JSONResponse({
                "status": "acknowledged",
                "action": "set_controller",
                "controller": broadcaster.active_controller_name,
                "controller_info": broadcaster.controller_telemetry,
            })
        elif action == "set_routing_mode":
            revision = body.get("demand_revision")
            if revision is not None and (revision != broadcaster.demand_revision
                                         or body.get("comparison_controller") != broadcaster.active_controller_name):
                return JSONResponse({"message": "Demand or the signal controller changed. Run the comparison again."},
                                    status_code=409)
            mode = str(body.get("mode", "")).strip().upper()
            if mode not in ROUTING_MODES:
                return JSONResponse({
                    "status": "error",
                    "message": f"Invalid routing mode '{mode}'. Must be one of: {list(ROUTING_MODES.keys())}",
                }, status_code=400)
            telemetry = broadcaster.set_routing_mode(mode)
            return JSONResponse({
                "status": "acknowledged",
                "action": "set_routing_mode",
                "mode": broadcaster.active_routing_mode,
                "active_mode": broadcaster.active_routing_mode,
                "routing_info": telemetry,
            })
        elif action == "load_scenario":
            scen = str(body.get("scenario", "normal")).lower()
            await broadcaster.handle_client_message(body)
            return JSONResponse({
                "status": "acknowledged",
                "action": "load_scenario",
                "scenario": scen,
            })
        result = await broadcaster.handle_client_message(body)
        if action == "start_routing_demo":
            return JSONResponse({"status": "acknowledged", "action": action, "demo": result})
        return JSONResponse({"status": "acknowledged", "action": body.get("action")})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


async def websocket_endpoint(websocket: WebSocket):
    """Real-time bidirectional WebSocket stream for 3D twin."""
    await broadcaster.register(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                await broadcaster.handle_client_message(msg)
            except Exception:
                pass
    except WebSocketDisconnect:
        broadcaster.unregister(websocket)
    except Exception:
        broadcaster.unregister(websocket)


routes = [
    Route("/", index, methods=["GET"]),
    Route("/api/rl-benchmarks/{scenario}/csv", api_rl_benchmark_csv, methods=["GET"]),
    Route("/api/rl-benchmarks", api_rl_benchmarks, methods=["GET"]),
    Route("/api/status", api_status, methods=["GET"]),
    Route("/api/controllers", api_controllers, methods=["GET"]),
    Route("/api/scenarios", api_scenarios, methods=["GET"]),
    Route("/api/observed-counts.csv", api_observed_counts, methods=["GET"]),
    Route("/api/demand/export.csv", api_export_demand, methods=["GET"]),
    Route("/api/routing", api_routing, methods=["GET"]),
    Route("/api/routing", api_set_routing, methods=["POST"]),
    Route("/api/demand", api_demand, methods=["POST"]),
    Route("/api/classroom-comparison-progress", api_classroom_comparison_progress, methods=["GET"]),
    Route("/api/routing-comparison", api_routing_comparison, methods=["GET"]),
    Route("/api/classroom-comparison", api_classroom_comparison, methods=["GET"]),
    Route("/api/experiments/matched", api_matched_comparison, methods=["GET"]),
    Route("/api/learned-controllers", api_learned_controllers, methods=["GET"]),
    Route("/api/experiments/held-out", api_held_out, methods=["GET"]),
    Route("/api/topology", api_topology, methods=["GET"]),
    Route("/api/route-plan", api_plan_route, methods=["POST"]),
    Route("/api/experiments/route", api_route_experiment, methods=["POST"]),
    Route("/api/control", api_control, methods=["POST"]),
    WebSocketRoute("/ws/live", websocket_endpoint),
    Mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static"),
]

app = Starlette(debug=True, routes=routes, lifespan=lifespan)
