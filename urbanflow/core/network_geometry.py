"""Geometry shared by junction connectors and directed road centerlines."""
import math

from urbanflow.core.types import Vec3


def offset_point(points, index, offset):
    """Offset a polyline vertex along a bounded miter."""
    normals = []
    for start, end in ((index - 1, index), (index, index + 1)):
        if 0 <= start and end < len(points):
            dx = points[end].x - points[start].x
            dy = points[end].y - points[start].y
            length = math.hypot(dx, dy)
            if length > 1e-8:
                normals.append((-dy / length, dx / length))
    point = points[index]
    if not normals:
        return point
    nx, ny = normals[0]
    if len(normals) == 2:
        mx, my = nx + normals[1][0], ny + normals[1][1]
        scale = mx * nx + my * ny
        if scale > 0.5:
            nx, ny = mx / scale, my / scale
    return Vec3(point.x + nx * offset, point.y + ny * offset, point.z)


def centerline_length(points):
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(points, points[1:]))


def trim_start(points, distance):
    for index, (a, b) in enumerate(zip(points, points[1:])):
        length = math.hypot(b.x - a.x, b.y - a.y)
        if length > distance:
            fraction = distance / length
            start = Vec3(a.x + (b.x - a.x) * fraction, a.y + (b.y - a.y) * fraction, a.z)
            return [start, *points[index + 1:]]
        distance -= length
    return points


class VehicleOccupancy:
    """Track oriented vehicle bodies in nearby cells on irregular map geometry."""

    WIDTHS = {"sedan": 1.9, "van": 2.1, "bus": 2.6, "emergency": 2.0}
    CELL_SIZE = 8.0

    def __init__(self, vehicles):
        self.cells = {}
        self.locations = {}
        for vehicle in vehicles:
            self.update(vehicle)

    def _key(self, vehicle):
        return math.floor(vehicle.x / self.CELL_SIZE), math.floor(vehicle.y / self.CELL_SIZE)

    def _body(self, vehicle):
        heading = math.radians(vehicle.heading)
        forward = math.sin(heading), math.cos(heading)
        side = forward[1], -forward[0]
        return (vehicle.x, vehicle.y, forward, side, vehicle.length / 2,
                self.WIDTHS.get(vehicle.type, 1.9) / 2)

    def update(self, vehicle):
        key = self._key(vehicle)
        prior = self.locations.get(vehicle.id)
        if prior is not None and prior != key:
            self.cells[prior].pop(vehicle.id)
        self.cells.setdefault(key, {})[vehicle.id] = self._body(vehicle)
        self.locations[vehicle.id] = key

    def blocked(self, vehicle):
        x, y, forward, side, half_length, half_width = self._body(vehicle)
        col, row = self._key(vehicle)
        for c in range(col - 2, col + 3):
            for r in range(row - 2, row + 3):
                for other, (ox, oy, of, os, ol, ow) in self.cells.get((c, r), {}).items():
                    if other == vehicle.id:
                        continue
                    dx, dy = ox - x, oy - y
                    limit = math.hypot(half_length, half_width) + math.hypot(ol, ow)
                    if dx * dx + dy * dy >= limit * limit:
                        continue
                    axes = forward, side, of, os
                    spans = half_length, half_width, ol, ow
                    if all(abs(dx * ax + dy * ay) <
                           sum(span * abs(ax * bx + ay * by) for span, (bx, by) in zip(spans, axes))
                           for ax, ay in axes):
                        return True
        return False
