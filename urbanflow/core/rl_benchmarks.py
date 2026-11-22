"""Read archived SUMO evaluation runs separately from UrbanFlow replays."""
import csv
from math import isfinite
from pathlib import Path
from statistics import mean

from urbanflow.core.observed_demand import COUNT_SOURCE, OBSERVED_WINDOWS


ARCHIVE = Path(__file__).resolve().parents[2] / "data" / "rl_benchmarks"
PRESET_NAMES = {"normal": "Normal", "rush_hour": "Rush hour", "surge": "Peak demand",
                "city_shift": "Commuter reversal", "event_release": "Event release"}
PRESET_STUDIES = {**{key: f"preset_{key}" for key in PRESET_NAMES}, **{key: key for key in OBSERVED_WINDOWS}}
SCENARIOS = {
    **{key: f"NYC DOT counts | Jan {day}, 2013 | {hour:02}:00" for key, (day, hour, _) in OBSERVED_WINDOWS.items()},
    **{f"preset_{key}": f"Synthetic reference | {name}" for key, name in PRESET_NAMES.items()},
    "single_test_shift": "Single intersection | Demand shift",
    "single_test_asym_450_250": "Single intersection | 450 / 250 asymmetry",
    "single_test_asym_500_200": "Single intersection | 500 / 200 asymmetry",
    "single_test_sym350": "Single intersection | Symmetric 350",
    "real_all_test_real_evening": "Profile | Evening",
    "real_all_test_real_midday": "Profile | Midday",
    "real_all_test_real_night": "Profile | Night",
    "real_all_test_real_profile": "Profile | Full demand",
    "grid_test_sym": "Grid | Symmetric demand",
}


def read_benchmarks(keys=SCENARIOS):
    studies = []
    for key in keys:
        title = SCENARIOS[key]
        observations = {}
        seen = set()
        with (ARCHIVE / f"{key}.csv").open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"controller", "seed", "avg_time_loss", "avg_depart_delay", "arrived"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{key}: missing required archive fields")
            for row in reader:
                identity = (row["controller"], row["seed"])
                if not all(identity) or identity in seen:
                    raise ValueError(f"{key}: missing or duplicate controller/seed")
                seen.add(identity)
                if not all(isfinite(float(row[field])) for field in required - {"controller", "seed"}):
                    raise ValueError(f"{key}: invalid archive metric")
                observations.setdefault(row["controller"], []).append(row)
        if not observations:
            raise ValueError(f"{key}: empty archive")
        common_seeds = set.intersection(*(set(row["seed"] for row in rows) for rows in observations.values()))
        if not common_seeds:
            raise ValueError(f"{key}: no shared seeds")
        scores = []
        for controller, rows in observations.items():
            paired = [row for row in rows if row["seed"] in common_seeds]
            delay = [float(row["avg_time_loss"]) + float(row["avg_depart_delay"]) for row in paired]
            scores.append({"controller": controller, "is_rl": controller.upper().startswith(("PPO", "IPPO", "MAPPO")),
                           "delay": mean(delay), "completed": mean(float(row["arrived"]) for row in paired)})
        scores.sort(key=lambda row: row["delay"])
        measured = key in OBSERVED_WINDOWS
        context = ("Measured directional counts and simulated trips" if measured else
                   "Synthetic preset departures and simulated turns" if key.startswith("preset_") else "Archived SUMO benchmark")
        studies.append({"id": key, "name": title, "source": f"{key}.csv", "seed_count": len(common_seeds), "results": scores,
                        "context": context, "engine": "SUMO", "seeds": sorted(common_seeds),
                        "duration_seconds": 3600 if measured else 600 if key.startswith("preset_") else None,
                        "source_url": COUNT_SOURCE if measured else None})
    return {"studies": studies}
