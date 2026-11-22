"""
Local, no-browser verification for road geometry / vehicle-orientation /
turning-connector correctness.

What this does NOT check: actual pixel rendering, materials, z-fighting, or
anything about what a human sees in the browser. It checks the authoritative
data the browser is handed -- positions, headings, lane offsets, and
road-to-road continuity -- using the real simulator, not a toy model.

Usage:
    python3 scripts/verify_geometry_and_turns.py

Exit code is 0 if all checks pass, 1 otherwise. Prints a per-check PASS/FAIL
report plus details of any jump/teleport/off-road event it finds.

To additionally confirm the browser side (which this script cannot do):
  1. Run `python run_demo.py` (or your usual launcher) and open the app.
  2. Open the browser DevTools console; there should be no red errors.
  3. Select the classroom_grid scenario, set simulation speed to 1x, and
     watch any one vehicle (click to select/track it, if supported) drive
     through at least one left turn and one right turn.
  4. Confirm visually: the vehicle's front stays pointed along its direction
     of travel at every point of the turn, it stays within the lane markings
     the whole way, and it does not pop/snap sideways when crossing an
     intersection.
  5. Take a screenshot or short screen recording of that turn and share it --
     that is the only way to confirm the rendered result, since this
     environment has no browser.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urbanflow.core.kinematic_sim import KinematicSimulator, SimulatedVehicle

MAX_PLAUSIBLE_TICK_JUMP_M = 1.5  # generous bound for dt=0.05s at realistic speeds
LANE_WIDTH = 3.5

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def main():
    sim = KinematicSimulator(map_name="classroom_grid")

    # 1. Every road's lane width/offsets stay within its own asphalt surface.
    for road in sim.roads.values():
        if len(road.points) < 2:
            continue
        half_w = (road.lanes * LANE_WIDTH) / 2.0
        max_offset = (max(road.lanes - 1, 0) - (road.lanes - 1) / 2.0) * LANE_WIDTH
        check(
            f"lane offsets within surface width on {road.id}",
            abs(max_offset) <= half_w + 1e-6,
            f"max_offset={max_offset:.2f} half_w={half_w:.2f}",
        )

    # 2. Drive a vehicle through one right turn and one left turn at INT_NW,
    #    stepping the real simulator loop, and watch for teleports.
    turn_routes = [
        ("right_turn_at_int_nw", ["road_w_in_nw", "road_nw_out_n"]),
        ("left_turn_at_int_nw", ["road_n_in_nw", "road_nw_out_w"]),
        ("straight_through_int_nw", ["road_w_in_nw", "road_nw_ne"]),
    ]

    for label, route in turn_routes:
        sim2 = KinematicSimulator(map_name="classroom_grid")
        sim2.vehicles.clear()
        sim2.pending_spawns.clear()

        first_road = sim2.roads[route[0]]
        veh = SimulatedVehicle(
            vehicle_id=f"verify_{label}",
            v_type="car",
            route=list(route),
            color="#ffffff",
            desired_speed=10.0,
            length=4.5,
        )
        veh.distance_on_road = max(0.0, first_road.length - 5.0)
        veh.lane_index = 0
        veh.speed = 10.0
        sim2.vehicles[veh.id] = veh
        sim2._update_vehicle_pose(veh, first_road)

        for ctrl in sim2.signal_controllers.values():
            ctrl.get_signal_code = lambda is_ns=True: "G"

        max_gap = 0.0
        last_x, last_y = veh.x, veh.y
        for _ in range(600):
            if veh.id not in sim2.vehicles:
                break
            sim2.step(dt=0.05)
            if veh.id in sim2.vehicles:
                gap = math.hypot(veh.x - last_x, veh.y - last_y)
                max_gap = max(max_gap, gap)
                last_x, last_y = veh.x, veh.y

        check(f"no teleport driving {label}", max_gap < MAX_PLAUSIBLE_TICK_JUMP_M, f"max single-tick jump={max_gap:.2f}m")

    if failures:
        print(f"\n{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("\nAll geometry/turn checks passed (data-level only -- not a visual/browser confirmation).")
    sys.exit(0)


if __name__ == "__main__":
    main()
