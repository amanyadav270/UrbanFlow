"""Validation and reproducible demand schedules for the classroom grid."""
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import io
import json
import math
import random

import networkx as nx

from urbanflow.core.observed_demand import DIRECTION, OBSERVED_WINDOWS, count_events


FLEET = {
    "sedan": (4.2, 12.0, "#2563eb"),
    "van": (5.4, 11.0, "#f59e0b"),
    "bus": (10.5, 9.5, "#059669"),
    "emergency": (5.0, 15.0, "#ef4444"),
}
SCENARIOS = {
    "normal": (1.6, 120),
    "rush_hour": (0.65, 160),
    "surge": (0.35, 160),
    "lane_closure": (0.65, 160),
    "incident": (0.65, 160),
    "gridlock": (0.35, 160),
}
STRESS_SCENARIOS = {
    "city_shift": ("Synthetic commuter reversal", ((0.9, 105, "east"), (0.6, 105, "west"), (1.3, 90, None))),
    "event_release": ("Synthetic event release", ((2.4, 75, None), (0.33, 90, "south"), (1.1, 135, None))),
}
TRAFFIC_LEVELS = (1, 1.5, 2, 3, 4)
DEMAND_FIELDS = ("vehicle_id", "departure_time", "entry_road", "destination", "entry_lane", "vehicle_type", "desired_speed", "route")


@dataclass(frozen=True)
class Departure:
    id: str
    time: float
    route: tuple[str, ...]
    lane: int
    kind: str
    speed: float


def classroom_demand(sim, scenario, seed=42):
    if scenario in OBSERVED_WINDOWS or scenario in STRESS_SCENARIOS:
        return directional_demand(sim, scenario, seed)
    if scenario not in SCENARIOS:
        raise ValueError("Unknown classroom demand scenario.")
    interval, duration = SCENARIOS[scenario]
    rng = random.Random(seed)
    routes = sim.scheduled_routes or sim.route_pool
    east_west = [route for route in routes if route[0].startswith(("road_w_", "road_e_"))]
    if not east_west:
        east_west = [
            route for route in routes
            if abs(sim.roads[route[0]].points[-1].x - sim.roads[route[0]].points[0].x)
            > abs(sim.roads[route[0]].points[-1].y - sim.roads[route[0]].points[0].y)
        ] or routes
    records = []
    for i in range(int(duration / interval)):
        # The first part is free flow. Rush profiles then concentrate demand on east-west approaches.
        time = i * interval if scenario == "normal" or i >= 15 else i * 1.3
        if scenario != "normal" and i >= 15:
            time = 19.5 + (i - 15) * interval
        pool = east_west if scenario != "normal" and i >= 15 and rng.random() < 0.8 else routes
        route = rng.choice(pool)
        kind = rng.choices(list(FLEET), (76, 14, 8, 2))[0]
        if sim.map_name == "bronx_fordham" or not sim.route_fits_vehicle(route, FLEET[kind][0]):
            kind = "sedan"
        records.append(Departure(f"demand-{i+1}", time, tuple(route), rng.randrange(sim.roads[route[0]].lanes),
                                 kind, FLEET[kind][1]))
    return tuple(records)


def directional_demand(sim, scenario, seed):
    rng = random.Random(seed + 1_000_000)
    if scenario in OBSERVED_WINDOWS:
        events = [(time, DIRECTION[direction]) for time, direction in count_events(scenario, seed)]
    else:
        events, elapsed = [], 0
        for interval, duration, side in STRESS_SCENARIOS[scenario][1]:
            events.extend((elapsed + (i + rng.random()) * interval, side if rng.random() < 0.82 else None)
                          for i in range(int(duration / interval)))
            elapsed += duration
    return project_events(sim, events, scenario, rng)


def project_events(sim, events, scenario, rng):
    """Project directional traffic events onto the selected network's entrances."""
    routes = sim.scheduled_routes or sim.route_pool
    pools = {side: [] for side in DIRECTION.values()}
    for route in routes:
        first = sim.roads[route[0]].points
        dx, dy = first[-1].x - first[0].x, first[-1].y - first[0].y
        side = ("west" if dx > 0 else "east") if abs(dx) > abs(dy) else ("south" if dy > 0 else "north")
        pools[side].append(route)
    records = []
    for index, (time, side) in enumerate(events, 1):
        route = rng.choice(pools.get(side) or routes)
        kind = "sedan"
        if scenario in STRESS_SCENARIOS and sim.map_name != "bronx_fordham":
            candidate = rng.choices(list(FLEET), (72, 14, 12, 2))[0]
            if sim.route_fits_vehicle(route, FLEET[candidate][0]):
                kind = candidate
        records.append(Departure(f"{scenario}-{index}", time, tuple(route),
                                 rng.randrange(sim.roads[route[0]].lanes), kind, FLEET[kind][1]))
    return tuple(records)


def scale_departures(sim, base, factor):
    if factor == 1:
        return base
    target = round(len(base) * factor)
    if target > 10000:
        raise ValueError("The scaled schedule exceeds 10,000 vehicles. Choose a lower traffic level.")
    rng = random.Random(42)
    records = list(base)
    identifiers = {record.id for record in base}
    source = list(base) * (int(factor) - 1)
    source += rng.sample(base, target - len(base) - len(source))
    serial = 0
    for record in source:
        serial += 1
        while f"traffic-copy-{serial}" in identifiers:
            serial += 1
        lanes = sim.roads[record.route[0]].lanes
        records.append(replace(record, id=f"traffic-copy-{serial}",
                               time=min(86400, record.time + rng.uniform(0.1, 3)),
                               lane=(record.lane + 1) % lanes))
    return tuple(sorted(records, key=lambda record: record.time))


def write_demand_csv(departures):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(DEMAND_FIELDS)
    for record in departures:
        writer.writerow((record.id, record.time, record.route[0], record.route[-1], record.lane, record.kind,
                         record.speed, json.dumps(record.route, separators=(",", ":"))))
    return stream.getvalue()


def parse_count_rows(sim, rows):
    counts = {}
    total = 0
    for number, row in enumerate(rows, 1):
        try:
            start = datetime(*(int(row[key]) for key in ("yr", "m", "d", "hh", "mm")))
            count = int(row["vol"])
            direction = DIRECTION[row["direction"].strip().upper()]
            if start.minute % 15 or count < 0 or (start, direction) in counts:
                raise ValueError
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValueError(f"Count row {number}: use a unique 15-minute slot per direction with a nonnegative count.") from exc
        counts[start, direction] = count
        total += count
    if not 1 <= total <= 10000:
        raise ValueError("The counts must represent between 1 and 10,000 vehicles.")
    # Replay each recorded hour in sequence, preserving quarter-hour spacing within it.
    hours = sorted({start.replace(minute=0) for start, _ in counts})
    offsets = {hour: index * 3600 for index, hour in enumerate(hours)}
    if len(hours) > 24:
        raise ValueError("Upload at most 24 recorded hours at a time.")
    rng = random.Random(42)
    events = sorted((offsets[start.replace(minute=0)] + start.minute * 60 + rng.random() * 900, direction)
                    for (start, direction), count in sorted(counts.items()) for _ in range(count))
    return project_events(sim, events, "uploaded", rng)


def parse_demand(sim, content, format_name):
    if format_name == "csv":
        reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
        reader.fieldnames = [field.strip().lower() for field in reader.fieldnames or ()]
        headers = set(reader.fieldnames)
        rows = list(reader)
        if {"yr", "m", "d", "hh", "mm", "vol", "direction"} <= headers:
            return parse_count_rows(sim, rows)
        if not set(DEMAND_FIELDS[:4]) <= headers:
            raise ValueError("Upload a departure schedule or a NYC DOT count CSV. Controller result CSVs contain scores only.")
    elif format_name == "json":
        rows = json.loads(content.lstrip("\ufeff"))
    else:
        raise ValueError("Choose a CSV or JSON file.")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("Provide between 1 and 10,000 vehicle records.")
    result = []
    seen = set()
    for number, row in enumerate(rows, 1):
        try:
            if not isinstance(row, dict):
                raise ValueError("Expected a vehicle record.")
            if not isinstance(row.get("vehicle_id"), (str, int)):
                raise ValueError("Vehicle ID must be text or an integer.")
            vehicle_id = str(row["vehicle_id"]).strip()
            if not vehicle_id or len(vehicle_id) > 80 or vehicle_id in seen:
                raise ValueError("Vehicle IDs must be unique and 1–80 characters long.")
            time = float(row["departure_time"])
            entry = sim.roads[str(row["entry_road"])]
            destination = sim.roads[str(row["destination"])]
            if entry.id.startswith("CONN__") or destination.id.startswith("CONN__"):
                raise ValueError("Use directed road IDs from the selected network.")
            if sim.map_name == "classroom_grid" and (
                entry.from_node in sim.intersections or destination.to_node in sim.intersections
            ):
                raise ValueError("Use a boundary entry road and a boundary exit road.")
            lane = int(str(row.get("entry_lane") if row.get("entry_lane") not in (None, "") else 0))
            kind = str(row.get("vehicle_type") or "sedan").lower()
            speed = float(row.get("desired_speed") or FLEET[kind][1])
            if not (math.isfinite(time) and 0 <= time <= 86400 and 0 <= lane < entry.lanes
                    and math.isfinite(speed) and 1 <= speed <= 45):
                raise ValueError("Check departure time, zero-based entry lane and desired speed in m/s.")
            if row.get("route"):
                route = tuple(json.loads(row["route"]) if isinstance(row["route"], str) else row["route"])
                if (not route or route[0] != entry.id or route[-1] != destination.id
                        or any(road.startswith("CONN__") or road not in sim.roads for road in route)
                        or any(sim.roads[a].to_node != sim.roads[b].from_node or sim.roads[a].from_node == sim.roads[b].to_node
                               for a, b in zip(route, route[1:]))):
                    raise ValueError("The saved route is invalid for this network.")
            elif entry.id == destination.id:
                route = (entry.id,)
            else:
                excluded = {(entry.to_node, entry.from_node), (destination.to_node, destination.from_node)}
                graph = nx.subgraph_view(sim.graph, filter_edge=lambda u, v: (u, v) not in excluded)
                nodes = nx.shortest_path(graph, entry.to_node, destination.from_node, weight="length")
                route = (entry.id, *(graph[a][b]["id"] for a, b in zip(nodes, nodes[1:])), destination.id)
            if len(route) == 2 and entry.from_node == destination.to_node:
                raise ValueError("Immediate U-turns are unavailable.")
            if sim.map_name == "bronx_fordham" and kind != "sedan":
                raise ValueError("The Bronx model supports sedan-length vehicles.")
            if not sim.route_fits_vehicle(route, FLEET[kind][0]):
                raise ValueError("This vehicle is too long for a link between junctions on its route.")
        except KeyError as exc:
            raise ValueError(f"Record {number}: unknown road or vehicle type {exc}. Export a schedule from the selected map.") from exc
        except (ValueError, TypeError, AttributeError, OverflowError, nx.NetworkXException) as exc:
            raise ValueError(f"Record {number}: {exc}") from exc
        seen.add(vehicle_id)
        result.append(Departure(vehicle_id, time, route, lane, kind, speed))
    return tuple(sorted(result, key=lambda record: record.time))
