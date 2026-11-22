"""UrbanFlow AI — Routing KPI Evaluator & Benchmark Metrics (Rev 6).

Implements Phase 4 Routing Performance Metrics according to the specification:
- Completed trip travel time, delay, waiting time, and network throughput.
- Incomplete trips accounting (strictly segregated from completed trips).
- Mean and peak queue lengths.
- Physical spillback event count and total duration.
- Total route changes and diversion rates.
- Operational V/C mean and peak.
- Traffic displacement / Vehicle-Kilometers Traveled (VKT) on alternative detour roads.

Defines explicit units, denominators, and warm-up handling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class RoutingKPISummary:
    """Summary of routing performance KPIs for a single simulation run."""
    # Metadata
    run_id: str
    scenario_family: str
    regime: str  # 'baseline', 'reactive', 'predictive'
    seed: int
    duration_sec: float
    warmup_sec: float
    eval_duration_sec: float

    # Trip Volumes & Completion
    total_spawned_trips: int
    completed_trips_count: int
    incomplete_trips_count: int
    warmup_completed_trips_count: int
    completion_rate: float
    network_throughput_veh_hr: float

    # Completed Trips Metrics (Post-Warmup Only)
    mean_travel_time_sec: float
    median_travel_time_sec: float
    p95_travel_time_sec: float
    mean_delay_sec: float
    mean_waiting_time_sec: float
    total_completed_travel_time_sec: float
    total_completed_wait_time_sec: float

    # Incomplete Trips Accounting (Strictly Segregated)
    incomplete_active_vehicles: int
    incomplete_accumulated_wait_time_sec: float
    incomplete_mean_distance_progress_m: float
    total_network_wait_time_sec: float

    # Queue Metrics (Post-Warmup)
    mean_queue_length_m: float
    p95_queue_length_m: float
    max_queue_length_m: float

    # Physical Spillback Metrics (Post-Warmup)
    spillback_event_count: int
    total_spillback_duration_sec: float
    mean_spillback_duration_sec: float
    max_spillback_duration_sec: float

    # Rerouting Activity & Diversion
    total_route_changes: int
    unique_diverted_vehicles_count: int
    overall_diversion_rate: float
    completed_diversion_rate: float

    # Operational V/C (Post-Warmup)
    mean_operational_vc: float
    p95_operational_vc: float
    peak_operational_vc: float

    # Traffic Displacement / Alternative VKT
    total_vkt_km: float
    alternative_vkt_km: float
    alternative_vkt_ratio: float

    # Counterfactual Spawn & Entry Metrics
    mean_system_travel_time_sec: float = 0.0
    median_system_travel_time_sec: float = 0.0
    p95_system_travel_time_sec: float = 0.0
    mean_network_travel_time_sec: float = 0.0
    mean_entry_delay_sec: float = 0.0
    max_entry_delay_sec: float = 0.0
    total_entry_delay_sec: float = 0.0
    never_entered_trips_count: int = 0
    total_system_wait_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RoutingKPIEvaluator:
    """Evaluates microscopic simulation telemetry and completed trip records into standardized KPIs."""

    DEFAULT_WARMUP_SEC: float = 600.0  # First 10 minutes excluded from steady-state statistics
    DEFAULT_SPILLBACK_RATIO: float = 0.90  # Queue >= 90% of segment length constitutes physical spillback

    def __init__(
        self,
        roads: Optional[Dict[str, Any]] = None,
        alternative_roads: Optional[Set[str]] = None,
        warmup_sec: float = DEFAULT_WARMUP_SEC,
        spillback_occupancy_threshold: float = DEFAULT_SPILLBACK_RATIO,
    ):
        self.roads = roads or {}
        self.alternative_roads = set(alternative_roads) if alternative_roads else self._auto_detect_alternative_roads()
        self.warmup_sec = float(warmup_sec)
        self.spillback_threshold = float(spillback_occupancy_threshold)

    def _auto_detect_alternative_roads(self) -> Set[str]:
        """Identifies known secondary / detour corridors in Bronx-Fordham network."""
        # Known alternative branching roads connecting between Grand Concourse and Webster / Washington Ave
        known_alternatives = {
            "road_5698666_15_fwd",
            "road_1082220764_167_fwd",
            "road_223889617_62_fwd",
            "road_223889617_63_fwd",
            "road_223889618_64_fwd",
            "road_223889618_65_fwd",
            "road_1082220766_170_fwd",
            "road_1082220766_171_fwd",
        }
        return {r for r in known_alternatives if r in self.roads} if self.roads else known_alternatives

    def calculate_free_flow_time(self, route: List[str]) -> float:
        """Calculates free-flow travel time (seconds) across a sequence of roads."""
        t_free = 0.0
        for r_id in route:
            road = self.roads.get(r_id)
            if road:
                length = getattr(road, "length", None) or (road.get("length") if isinstance(road, dict) else 100.0)
                speed = getattr(road, "speed_limit", None) or (road.get("speed_limit") if isinstance(road, dict) else 11.2)
                t_free += float(length) / max(0.1, float(speed))
            else:
                t_free += 10.0  # default fallback
        return max(0.1, t_free)

    def calculate_route_length_m(self, route: List[str]) -> float:
        """Calculates cumulative road length (meters) across a sequence of roads."""
        length_m = 0.0
        for r_id in route:
            road = self.roads.get(r_id)
            if road:
                l = getattr(road, "length", None) or (road.get("length") if isinstance(road, dict) else 100.0)
                length_m += float(l)
            else:
                length_m += 100.0
        return length_m

    def calculate_vkt_km(
        self,
        trips: List[Dict[str, Any]],
        filter_roads: Optional[Set[str]] = None,
    ) -> float:
        """Calculates Vehicle-Kilometers Traveled (VKT) in km for a set of trips."""
        total_km = 0.0
        for trip in trips:
            route = trip.get("final_route") or trip.get("route") or []
            if filter_roads:
                # Sum lengths only for roads in filter_roads
                dist_m = sum(
                    (getattr(self.roads[r], "length", 100.0) if hasattr(self.roads.get(r), "length")
                     else (self.roads[r].get("length", 100.0) if isinstance(self.roads.get(r), dict) else 100.0))
                    for r in route if r in filter_roads and r in self.roads
                )
            else:
                dist_m = trip.get("distance_traveled_m")
                if dist_m is None:
                    dist_m = self.calculate_route_length_m(route)
            total_km += float(dist_m) / 1000.0
        return round(total_km, 4)

    def detect_spillback_events(
        self,
        telemetry: List[Dict[str, Any]],
        sample_interval_sec: float = 5.0,
    ) -> Tuple[int, float, float, float]:
        """Detects physical spillback episodes across road segments post-warmup.

        A road segment enters spillback when:
            queue_length_m >= spillback_threshold * road_length
        Contiguous timestamps where this condition holds form a single event.

        Returns:
            Tuple of:
                - event_count (int)
                - total_duration_sec (float)
                - mean_duration_sec (float)
                - max_duration_sec (float)
        """
        if not telemetry:
            return 0, 0.0, 0.0, 0.0

        # Filter post-warmup observations
        post_warmup = [obs for obs in telemetry if obs.get("timestamp", 0.0) >= self.warmup_sec]
        if not post_warmup:
            return 0, 0.0, 0.0, 0.0

        # Group by road_id
        by_road: Dict[str, List[Dict[str, Any]]] = {}
        for obs in post_warmup:
            r_id = obs.get("road_id")
            if r_id:
                by_road.setdefault(r_id, []).append(obs)

        event_durations: List[float] = []

        for r_id, obs_list in by_road.items():
            road = self.roads.get(r_id)
            if road:
                road_len = getattr(road, "length", None) or (road.get("length") if isinstance(road, dict) else 100.0)
            else:
                road_len = 100.0
            road_len = float(road_len)

            # Critical threshold
            thresh_m = max(10.0, min(road_len * self.spillback_threshold, road_len - 15.0))

            obs_sorted = sorted(obs_list, key=lambda x: x.get("timestamp", 0.0))

            in_spillback = False
            cur_duration = 0.0
            last_ts = -1.0

            for obs in obs_sorted:
                ts = float(obs.get("timestamp", 0.0))
                q_len = float(obs.get("queue_length_m") or obs.get("queue_length_meters") or 0.0)

                is_spill = (q_len >= thresh_m)

                if is_spill:
                    if not in_spillback:
                        in_spillback = True
                        cur_duration = sample_interval_sec
                    else:
                        dt = ts - last_ts if last_ts >= 0 else sample_interval_sec
                        cur_duration += dt if (0 < dt <= sample_interval_sec * 2.5) else sample_interval_sec
                else:
                    if in_spillback:
                        event_durations.append(cur_duration)
                        in_spillback = False
                        cur_duration = 0.0
                last_ts = ts

            if in_spillback and cur_duration > 0.0:
                event_durations.append(cur_duration)

        count = len(event_durations)
        total_dur = float(sum(event_durations))
        mean_dur = float(np.mean(event_durations)) if count > 0 else 0.0
        max_dur = float(max(event_durations)) if count > 0 else 0.0

        return count, round(total_dur, 2), round(mean_dur, 2), round(max_dur, 2)

    def calculate_kpis(
        self,
        completed_trips: List[Dict[str, Any]],
        active_vehicles: List[Dict[str, Any]],
        telemetry: List[Dict[str, Any]],
        run_id: str = "RUN_BENCHMARK",
        scenario_family: str = "FAM-05",
        regime: str = "reactive",
        seed: int = 5001,
        duration_sec: float = 3600.0,
        sample_interval_sec: float = 5.0,
        never_entered_vehicles: Optional[List[Dict[str, Any]]] = None,
    ) -> RoutingKPISummary:
        """Computes comprehensive routing KPIs across completed trips, active fleet, telemetry, and pending entry queue."""
        eval_duration_sec = max(1.0, duration_sec - self.warmup_sec)

        # 1. Segregate completed trips by warm-up threshold
        post_warmup_completed: List[Dict[str, Any]] = []
        warmup_completed: List[Dict[str, Any]] = []

        for trip in completed_trips:
            arr_time = float(trip.get("arrival_time") if trip.get("arrival_time") is not None else trip.get("completed_at", 0.0))
            if arr_time >= self.warmup_sec:
                post_warmup_completed.append(trip)
            else:
                warmup_completed.append(trip)

        n_completed = len(post_warmup_completed)
        n_warmup_completed = len(warmup_completed)
        n_incomplete = len(active_vehicles)
        never_ent_list = never_entered_vehicles or []
        n_never_entered = len(never_ent_list)
        n_total_spawned = n_completed + n_warmup_completed + n_incomplete + n_never_entered

        # Eligible post-warmup demand denominator
        post_warmup_spawned = n_completed + n_incomplete + n_never_entered
        completion_rate = (n_completed / max(1, post_warmup_spawned)) if post_warmup_spawned > 0 else 0.0
        throughput_veh_hr = (n_completed / eval_duration_sec) * 3600.0

        # 2. Travel Time & Delay (Completed Trips Post-Warmup)
        travel_times = []
        network_travel_times = []
        system_travel_times = []
        entry_delays = []
        delays = []
        wait_times = []

        for trip in post_warmup_completed:
            t_total = float(trip.get("travel_time_sec") if trip.get("travel_time_sec") is not None else trip.get("travel_time", 0.0))
            t_net = float(trip.get("network_travel_time_sec") if trip.get("network_travel_time_sec") is not None else t_total)
            t_entry_delay = float(trip.get("entry_delay_sec", 0.0))
            t_sys = float(trip.get("system_travel_time_sec") if trip.get("system_travel_time_sec") is not None else (t_net + t_entry_delay))

            t_wait = float(trip.get("wait_time_sec") if trip.get("wait_time_sec") is not None else trip.get("wait_time", 0.0))
            f_route = trip.get("final_route") or []
            t_free = self.calculate_free_flow_time(f_route)
            delay = max(0.0, t_total - t_free)

            travel_times.append(t_total)
            network_travel_times.append(t_net)
            system_travel_times.append(t_sys)
            entry_delays.append(t_entry_delay)
            delays.append(delay)
            wait_times.append(t_wait)

        mean_tt = float(np.mean(travel_times)) if travel_times else 0.0
        median_tt = float(np.median(travel_times)) if travel_times else 0.0
        p95_tt = float(np.percentile(travel_times, 95)) if travel_times else 0.0
        mean_delay = float(np.mean(delays)) if delays else 0.0
        mean_wait = float(np.mean(wait_times)) if wait_times else 0.0
        total_comp_tt = float(sum(travel_times))
        total_comp_wait = float(sum(wait_times))

        # Counterfactual spawn & entry delay summaries
        mean_net_tt = float(np.mean(network_travel_times)) if network_travel_times else 0.0
        mean_sys_tt = float(np.mean(system_travel_times)) if system_travel_times else 0.0
        median_sys_tt = float(np.median(system_travel_times)) if system_travel_times else 0.0
        p95_sys_tt = float(np.percentile(system_travel_times, 95)) if system_travel_times else 0.0
        mean_entry_delay = float(np.mean(entry_delays)) if entry_delays else 0.0
        max_entry_delay = float(max(entry_delays)) if entry_delays else 0.0
        tot_entry_delay = float(sum(entry_delays))

        never_ent_delays = [float(v.get("entry_delay_sec", 0.0)) for v in never_ent_list]
        tot_never_ent_delay = float(sum(never_ent_delays))

        # 3. Incomplete Trips Accounting (Strictly Segregated)
        inc_waits = []
        inc_dists = []
        for v in active_vehicles:
            w = float(v.get("wait_time", 0.0) if isinstance(v, dict) else getattr(v, "wait_time", 0.0))
            d = float(v.get("distance_on_road", 0.0) if isinstance(v, dict) else getattr(v, "distance_on_road", 0.0))
            inc_waits.append(w)
            inc_dists.append(d)

        inc_total_wait = float(sum(inc_waits))
        inc_mean_dist = float(np.mean(inc_dists)) if inc_dists else 0.0
        total_net_wait = total_comp_wait + inc_total_wait
        total_sys_wait = total_net_wait + tot_entry_delay + tot_never_ent_delay

        # 4. Queue Metrics & Physical Spillback (Post-Warmup)
        pw_telem = [o for o in telemetry if float(o.get("timestamp", 0.0)) >= self.warmup_sec]
        if pw_telem:
            q_lens = [float(o.get("queue_length_m") or o.get("queue_length_meters") or 0.0) for o in pw_telem]
            vc_vals = [float(o.get("vc_ratio") or o.get("capacity_utilization") or 0.0) for o in pw_telem]
        else:
            q_lens = [0.0]
            vc_vals = [0.0]

        mean_q = float(np.mean(q_lens))
        p95_q = float(np.percentile(q_lens, 95))
        max_q = float(max(q_lens))

        spill_count, spill_total_dur, spill_mean_dur, spill_max_dur = self.detect_spillback_events(
            telemetry,
            sample_interval_sec=sample_interval_sec,
        )

        # 5. Rerouting & Diversion Activity
        all_trips = post_warmup_completed + (active_vehicles if isinstance(active_vehicles[0] if active_vehicles else {}, dict) else [
            {"reroute_count": getattr(v, "reroute_count", 0)} for v in active_vehicles
        ])
        total_reroutes = sum(int(t.get("reroute_count", 0)) for t in all_trips)
        unique_diverted = sum(1 for t in all_trips if int(t.get("reroute_count", 0)) > 0)

        overall_div_rate = (unique_diverted / max(1, n_total_spawned))
        completed_div_rate = (
            sum(1 for t in post_warmup_completed if int(t.get("reroute_count", 0)) > 0) / max(1, n_completed)
        )

        # 6. Operational V/C
        mean_vc = float(np.mean(vc_vals))
        p95_vc = float(np.percentile(vc_vals, 95))
        peak_vc = float(max(vc_vals))

        # 7. Traffic Displacement (VKT)
        total_vkt = self.calculate_vkt_km(post_warmup_completed)
        alt_vkt = self.calculate_vkt_km(post_warmup_completed, filter_roads=self.alternative_roads)
        alt_vkt_ratio = (alt_vkt / max(0.001, total_vkt)) if total_vkt > 0 else 0.0

        return RoutingKPISummary(
            run_id=run_id,
            scenario_family=scenario_family,
            regime=regime,
            seed=seed,
            duration_sec=duration_sec,
            warmup_sec=self.warmup_sec,
            eval_duration_sec=eval_duration_sec,
            total_spawned_trips=n_total_spawned,
            completed_trips_count=n_completed,
            incomplete_trips_count=n_incomplete,
            warmup_completed_trips_count=n_warmup_completed,
            completion_rate=round(completion_rate, 4),
            network_throughput_veh_hr=round(throughput_veh_hr, 2),
            mean_travel_time_sec=round(mean_tt, 2),
            median_travel_time_sec=round(median_tt, 2),
            p95_travel_time_sec=round(p95_tt, 2),
            mean_delay_sec=round(mean_delay, 2),
            mean_waiting_time_sec=round(mean_wait, 2),
            total_completed_travel_time_sec=round(total_comp_tt, 2),
            total_completed_wait_time_sec=round(total_comp_wait, 2),
            incomplete_active_vehicles=n_incomplete,
            incomplete_accumulated_wait_time_sec=round(inc_total_wait, 2),
            incomplete_mean_distance_progress_m=round(inc_mean_dist, 2),
            total_network_wait_time_sec=round(total_net_wait, 2),
            mean_queue_length_m=round(mean_q, 2),
            p95_queue_length_m=round(p95_q, 2),
            max_queue_length_m=round(max_q, 2),
            spillback_event_count=spill_count,
            total_spillback_duration_sec=spill_total_dur,
            mean_spillback_duration_sec=spill_mean_dur,
            max_spillback_duration_sec=spill_max_dur,
            total_route_changes=total_reroutes,
            unique_diverted_vehicles_count=unique_diverted,
            overall_diversion_rate=round(overall_div_rate, 4),
            completed_diversion_rate=round(completed_div_rate, 4),
            mean_operational_vc=round(mean_vc, 4),
            p95_operational_vc=round(p95_vc, 4),
            peak_operational_vc=round(peak_vc, 4),
            total_vkt_km=round(total_vkt, 3),
            alternative_vkt_km=round(alt_vkt, 3),
            alternative_vkt_ratio=round(alt_vkt_ratio, 4),
            mean_system_travel_time_sec=round(mean_sys_tt, 2),
            median_system_travel_time_sec=round(median_sys_tt, 2),
            p95_system_travel_time_sec=round(p95_sys_tt, 2),
            mean_network_travel_time_sec=round(mean_net_tt, 2),
            mean_entry_delay_sec=round(mean_entry_delay, 2),
            max_entry_delay_sec=round(max_entry_delay, 2),
            total_entry_delay_sec=round(tot_entry_delay, 2),
            never_entered_trips_count=n_never_entered,
            total_system_wait_time_sec=round(total_sys_wait, 2),
        )
