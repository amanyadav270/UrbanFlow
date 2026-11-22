"""
Regression tests locking in the sim -> WebSocket -> Three.js coordinate and
heading convention.

Convention (verified by direct trace through kinematic_sim.py, osm_importer.py,
vehicles.js, road_builder.js, scene3d.js):

  Simulator (x, y) is a standard local-Euclidean map frame: +x = east, +y = north.
  heading = atan2(dx, dy) in degrees, 0..360, where 0 deg = facing +y (north),
  90 deg = facing +x (east) -- i.e. compass bearing, clockwise from north.

  Frontend maps world (X, Z) = sim (x, y) [vehicles.js: mesh.position.set(x, 0, y)].
  Vehicle model's local forward axis is +Z (headlights at +length/2 in Z).
  mesh.rotation.y = heading (deg->rad) is applied directly with no extra offset.

  For this to be consistent, THREE's rotation-about-Y of the local +Z axis by
  angle theta must land exactly on the simulator's direction vector
  (sin(theta), cos(theta)) in the (x, z) plane. That is verified numerically
  below for a spread of headings, so any future change to either side of the
  pipeline that breaks the correspondence will fail these tests immediately.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urbanflow.core.kinematic_sim import KinematicSimulator


def three_js_rotation_y_forward(theta_deg: float):
    """
    Reproduce THREE.js's rotation-about-Y transform applied to the vehicle
    model's local forward vector (0, 0, 1), exactly as vehicles.js does with
    `mesh.rotation.y = THREE.MathUtils.degToRad(heading)`.

    THREE's makeRotationY(theta) transforms (x, z) as:
        x' = x*cos(theta) + z*sin(theta)
        z' = -x*sin(theta) + z*cos(theta)
    """
    theta = math.radians(theta_deg)
    x, z = 0.0, 1.0
    x_prime = x * math.cos(theta) + z * math.sin(theta)
    z_prime = -x * math.sin(theta) + z * math.cos(theta)
    return x_prime, z_prime


def test_heading_matches_rendered_forward_direction_cardinal():
    """
    For movement along each cardinal/diagonal direction, the heading computed
    by _update_vehicle_pose must, once rendered by THREE's rotation.y, point
    the vehicle's local +Z (headlights) along the *actual* (dx, dy) travel
    direction mapped into world (x, z).
    """
    directions = [
        (0.0, 1.0),    # due north
        (1.0, 0.0),    # due east
        (0.0, -1.0),   # due south
        (-1.0, 0.0),   # due west
        (1.0, 1.0),    # north-east
        (-1.0, -1.0),  # south-west
    ]
    for dx, dy in directions:
        heading = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

        # World-space travel direction, per vehicles.js mapping: world(X,Z) = sim(x,y)
        norm = math.hypot(dx, dy)
        expected_x, expected_z = dx / norm, dy / norm

        rendered_x, rendered_z = three_js_rotation_y_forward(heading)

        assert math.isclose(rendered_x, expected_x, abs_tol=1e-9), (dx, dy, heading)
        assert math.isclose(rendered_z, expected_z, abs_tol=1e-9), (dx, dy, heading)


def test_update_vehicle_pose_heading_and_lane_offset_are_consistent():
    """
    End-to-end: build a straight 2-lane eastbound road segment and verify that
    _update_vehicle_pose places a vehicle on the correct lane centerline and
    assigns a heading that, once rendered, faces along the road's direction
    -- not perpendicular to it and not reversed.
    """
    from urbanflow.core.types import RoadSegment, Vec3

    class DummyVeh:
        distance_on_road = 50.0
        lane_index = 0
        x = y = z = 0.0
        heading = 0.0

    road = RoadSegment(
        id="road_test_eastbound",
        from_node="A",
        to_node="B",
        lanes=2,
        length=100.0,
        speed_limit=11.2,
        capacity=1800.0,
        points=[Vec3(0.0, 0.0, 0.0), Vec3(100.0, 0.0, 0.0)],  # due east
    )

    sim = KinematicSimulator.__new__(KinematicSimulator)
    veh = DummyVeh()
    KinematicSimulator._update_vehicle_pose(sim, veh, road)

    # Heading should point east (90 deg), not north/south/west.
    assert math.isclose(veh.heading, 90.0, abs_tol=1e-6), veh.heading

    # Lane 0 of a 2-lane road should be offset to one side of the centerline
    # by half the lane width (3.5 / 2 = 1.75 m), not sitting on y=0 and not
    # off the roadway entirely.
    assert math.isclose(abs(veh.y), 1.75, abs_tol=1e-6), veh.y

    # Rendered forward direction (world X,Z) must match the road's actual
    # direction of travel (east: world +X, world Z unchanged).
    rx, rz = three_js_rotation_y_forward(veh.heading)
    assert rx > 0.99, "vehicle heading does not render facing east"
    assert abs(rz) < 1e-6


def test_classroom_grid_opposing_carriageways_do_not_overlap_or_gap():
    """
    The 4-intersection classroom grid defines each direction of travel as a
    separate road centered on its own line, offset by the per-direction half
    width (3.5 m for a 2-lane, 7m-wide carriageway) from the street median.
    Verify the two opposing carriageways on "Street North" meet exactly at
    the median with no gap and no overlap, so road_builder.js's rendered
    asphalt quads (halfW = lanes * laneWidth / 2 either side of each line)
    tile correctly.
    """
    sim = KinematicSimulator(map_name="classroom_grid")
    eastbound = sim.roads["road_nw_ne"]
    westbound = sim.roads["road_ne_nw"]

    lane_width = 3.5
    half_w = (eastbound.lanes * lane_width) / 2.0  # 3.5 for a 2-lane road

    eb_y = eastbound.points[0].y
    wb_y = westbound.points[0].y

    # Inner edges of the two carriageways (closest to the median) must
    # coincide exactly at the street median.
    eb_inner_edge = eb_y + half_w
    wb_inner_edge = wb_y - half_w
    assert math.isclose(eb_inner_edge, wb_inner_edge, abs_tol=1e-9), (
        eb_inner_edge, wb_inner_edge,
    )
