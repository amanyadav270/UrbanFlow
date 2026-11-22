"""Reproducible departure times derived from NYC DOT's 15-minute counts."""
import csv
from pathlib import Path
import random


COUNTS = Path(__file__).resolve().parents[2] / "data" / "traffic_counts" / "nyc-dot-bronx-2013.csv"
COUNT_SOURCE = "https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt"
OBSERVED_WINDOWS = {
    "dot_morning": (8, 7, "NYC DOT morning commute"),
    "dot_midday": (9, 12, "NYC DOT midday traffic"),
    "dot_evening": (10, 17, "NYC DOT evening commute"),
}
DIRECTION = {"NB": "south", "SB": "north", "EB": "west", "WB": "east"}


def measured_slots(window):
    day, hour, _ = OBSERVED_WINDOWS[window]
    with COUNTS.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if int(row["D"]) == day and int(row["HH"]) == hour]
    slots = {(int(row["MM"]), row["Direction"]): int(row["Vol"]) for row in rows}
    expected = {(minute, direction) for minute in (0, 15, 30, 45) for direction in DIRECTION}
    if len(rows) != len(expected) or slots.keys() != expected or min(slots.values()) < 0:
        raise ValueError(f"Incomplete or invalid NYC DOT count window: {window}")
    return slots


def count_events(window, seed=42):
    rng = random.Random(seed)
    return sorted((minute * 60 + rng.random() * 900, direction)
                  for (minute, direction), count in sorted(measured_slots(window).items())
                  for _ in range(count))
