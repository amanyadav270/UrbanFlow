"""Synthetic districts with directed lanes and plan-view junction envelopes."""
from itertools import combinations
import math
import random

import networkx as nx

from urbanflow.core.types import Intersection, RoadSegment, Vec3


DISTRICTS = {
    "midtown": ("Midtown avenues", [-160, 0, 160], [-130, 0, 130], [18, 30, 54], "#7f94b0"),
    "metro": ("Metro center", [-580, -360, -135, 90, 320, 550],
              [-535, -340, -100, 110, 335, 560], [15, 30, 54, 92], "#8192a9"),
    "harbor": ("Harbor city", [-790, -565, -360, -145, 70, 295, 530, 745],
               [-455, -280, -65, 105, 310, 510], [9, 16, 24, 42], "#9d9d8c"),
}
CITIES = {"metro", "harbor"}
AVAILABLE_NETWORKS = frozenset({"classroom_grid", "classroom", "classroom_demo", "bronx_fordham", "procedural", *DISTRICTS})
CITY_STOP_LINE_SETBACK = 4.0
CITY_CORRIDORS = {
    "metro": (["West End Ave", "Broadway", "Central Ave", "Union Ave", "Park Ave", "Eastway"],
              ["Museum St", "Civic Blvd", "Market St", "Library St", "Grand Blvd", "North Terrace"]),
    "harbor": (["West Wharf Rd", "Ferry Ave", "Canal Ave", "Beacon Ave", "Marina Ave", "Port Ave", "Shipyard Rd", "East Dock Rd"],
               ["Wharf St", "Harbor Blvd", "Customs St", "Pier St", "Seabreeze Blvd", "North Quay"]),
}


def build_district(sim, signal_class, key):
    name, xs, ys, heights, color = DISTRICTS[key]
    rng = random.Random(42)
    coords = {}
    corridor_names = CITY_CORRIDORS.get(key)

    def junction(i, j):
        return f"INT_{i + 1}_{j + 1}"

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            node = junction(i, j)
            coords[node] = (x, y)
            label = f"{corridor_names[0][i]} / {corridor_names[1][j]}" if corridor_names else None
            sim.intersections[node] = Intersection(node, Vec3(x, y, 0), [], name=label)
            sim.signal_controllers[node] = signal_class(node)

    boundaries = []
    links = []
    for i in range(len(xs)):
        column = [junction(i, j) for j in range(len(ys))]
        for side, y in (("south", ys[0] - 90), ("north", ys[-1] + 90)):
            node = f"{side}_{i + 1}"
            boundaries.append(node)
            coords[node] = (xs[i], y)
            if side == "south":
                column.insert(0, node)
            else:
                column.append(node)
        links.extend(zip(column, column[1:]))
    for j in range(len(ys)):
        row = [junction(i, j) for i in range(len(xs))]
        for side, x in (("west", xs[0] - 90), ("east", xs[-1] + 90)):
            node = f"{side}_{j + 1}"
            boundaries.append(node)
            coords[node] = (x, ys[j])
            if side == "west":
                row.insert(0, node)
            else:
                row.append(node)
        links.extend(zip(row, row[1:]))

    def add_road(start, end):
        x, y = coords[start]
        ex, ey = coords[end]
        length = math.hypot(ex - x, ey - y)
        tx, ty = (ex - x) / length, (ey - y) / length
        start_trim = 8 if start in sim.intersections else 0
        end_trim = 8 if end in sim.intersections else 0
        road_id = f"{start.lower()}__to__{end.lower()}"
        points = [Vec3(x + tx * start_trim + ty * 3.5, y + ty * start_trim - tx * 3.5, 0),
                  Vec3(ex - tx * end_trim + ty * 3.5, ey - ty * end_trim - tx * 3.5, 0)]
        length -= start_trim + end_trim
        speed = 13.89
        if key in CITIES and (x == ex == xs[2] or y == ey == ys[2]):
            speed = 16.67
        label = None
        if corridor_names:
            label = corridor_names[0][xs.index(x)] if x == ex else corridor_names[1][ys.index(y)]
        sim.roads[road_id] = RoadSegment(road_id, start, end, lanes=2, length=length,
                                         capacity=3600, speed_limit=speed, points=points, name=label)
        sim.graph.add_edge(start, end, id=road_id, length=length)
        for node in (start, end):
            if node in sim.intersections:
                sim.intersections[node].connected_roads.append(road_id)

    for a, b in links:
        add_road(a, b)
        add_road(b, a)

    for start, end in combinations(boundaries, 2):
        for a, b in ((start, end), (end, start)):
            path = nx.shortest_path(sim.graph, a, b, weight="length")
            sim.route_pool.append([sim.graph[u][v]["id"] for u, v in zip(path, path[1:])])

    surfaces = []
    for i, (x0, x1) in enumerate(zip(xs, xs[1:])):
        for j, (y0, y1) in enumerate(zip(ys, ys[1:])):
            park = key in CITIES and (i + 3 * j) % 9 == 2
            surfaces.append({"bounds": [x0 + 10, y0 + 10, x1 - 10, y1 - 10],
                             "kind": "park" if park else "block"})
            if park:
                continue
            city = key in CITIES
            warehouses = key == "harbor" and j == 0
            parcels = 2 if warehouses else rng.randint(2, 4) if city else 3
            for col in range(parcels):
                start = x0 + 17 + (x1 - x0 - 34) / parcels * col
                end = x0 + 17 + (x1 - x0 - 34) / parcels * (col + 1) - 5
                for low, high in ((y0 + 17, (y0 + y1) / 2 - 5), ((y0 + y1) / 2 + 5, y1 - 17)):
                    if city:
                        start_inset, end_inset = rng.randint(1, 8), rng.randint(1, 11)
                        low += rng.randint(1, 8)
                        high -= rng.randint(1, 12)
                        material = rng.choice(("#896e60", "#9b9890", "#58717f", "#b2a38a", "#78858c"))
                        height = rng.randint(9, 15) if warehouses else rng.choice(heights)
                    else:
                        start_inset = end_inset = 0
                        material, height = color, rng.choice(heights)
                    sim.buildings.append({"height": height, "color": material,
                        "city_facade": city and not warehouses,
                        "polygon": [{"x": x, "y": y} for x, y in
                                    ((start + start_inset, low), (end - end_inset, low),
                                     (end - end_inset, high), (start + start_inset, high))]})
    if key == "harbor":
        surfaces.append({"bounds": [xs[0] - 80, ys[0] - 180, xs[-1] + 80, ys[0] - 98], "kind": "water"})
    sim.map_metadata = {"name": name, "map_name": key, "junction_half_width": 8,
                        "surfaces": surfaces, "ground_size": max(1400, 400 + max(xs[-1] - xs[0], ys[-1] - ys[0]))}
    if key in CITIES:
        sim.map_metadata["stop_line_setback"] = CITY_STOP_LINE_SETBACK
        sim.map_metadata["cityscape"] = True
