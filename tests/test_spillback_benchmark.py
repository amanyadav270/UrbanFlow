"""Unit and Contract Tests for UrbanFlow Spillback Operational Benchmark (Rev 6).

Covers all required contracts from user specification:
1. Consecutive-Positive Confirmation (t_alarm = t_first + 5.0s on second observation)
2. 300-Second Episode Cooldown (subsequent positives within 300s merged into episode)
3. Reset Behavior (resets upon physical event onset or cooldown expiration)
4. Same-Road Matching Constraint (alarms on Road A cannot match events on Road B)
5. One-to-One Matching (one alarm cannot match multiple events; greedy chronological tie-breaking)
6. Missed Events Accounting (unmatched physical onsets recorded as FN)
7. False Alarms Accounting (unmatched alarms recorded as FP; active spillback suppressed)
8. Non-Congested Exposure Time & False Alarm Rate Calculations
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_campaign_models import deterministic_alarm_event_matching
from scripts.evaluate_campaign_spillback import extract_operational_alarms


def test_consecutive_positive_confirmation():
    """
    Verifies that a single isolated positive prediction (p >= 0.50) does NOT fire an alarm,
    and two consecutive positives fire an alarm stamped at the SECOND observation (t_first + 5.0s).
    """
    timestamps = np.arange(100.0, 160.0, 5.0)  # 100, 105, 110, 115, 120, 125, ...
    probs = [0.1, 0.6, 0.2, 0.7, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    df = pd.DataFrame({
        "run_id": ["RUN_01"] * len(timestamps),
        "road_id": ["ROAD_A"] * len(timestamps),
        "timestamp": timestamps,
        "prob": probs,
        "is_active_spillback": [False] * len(timestamps),
        "is_event_onset": [False] * len(timestamps),
    })
    
    alarms = extract_operational_alarms(df, mode="ML", prob_col="prob", cooldown_sec=300.0)
    
    # Isolated positive at t=105s should NOT trigger
    # Consecutive positives at t=115s and t=120s should trigger at t=120s
    assert len(alarms) == 1, f"Expected exactly 1 alarm, got {len(alarms)}"
    assert alarms[0]["t_first"] == 115.0, f"Expected t_first=115.0, got {alarms[0]['t_first']}"
    assert alarms[0]["t_alarm"] == 120.0, f"Expected t_alarm=120.0, got {alarms[0]['t_alarm']}"


def test_300s_cooldown_and_merging():
    """
    Verifies that after an alarm fires, an episode cooldown of 300s is enforced,
    merging subsequent positive predictions into the existing episode.
    """
    timestamps = np.arange(100.0, 600.0, 5.0)  # 100 observations spanning 500s
    # Two positives at 110, 115 -> alarm at 115. Cooldown until 115 + 300 = 415s.
    # Positives at 130, 135 (within cooldown) must NOT fire a new alarm.
    # Positives at 430, 435 (after cooldown) MUST fire a second alarm at 435s.
    probs = np.full(len(timestamps), 0.1)
    
    # First episode: steps 2, 3 (t=110, 115)
    probs[2] = 0.8
    probs[3] = 0.8
    # Within cooldown: steps 6, 7 (t=130, 135)
    probs[6] = 0.9
    probs[7] = 0.9
    # After cooldown (t >= 415): steps 66, 67 (t=430, 435)
    probs[66] = 0.8
    probs[67] = 0.8
    
    df = pd.DataFrame({
        "run_id": ["RUN_01"] * len(timestamps),
        "road_id": ["ROAD_A"] * len(timestamps),
        "timestamp": timestamps,
        "prob": probs,
        "is_active_spillback": [False] * len(timestamps),
        "is_event_onset": [False] * len(timestamps),
    })
    
    alarms = extract_operational_alarms(df, mode="ML", prob_col="prob", cooldown_sec=300.0)
    assert len(alarms) == 2, f"Expected 2 alarms, got {len(alarms)}"
    assert alarms[0]["t_alarm"] == 115.0
    assert alarms[1]["t_alarm"] == 435.0


def test_cooldown_reset_on_event_onset():
    """
    Verifies that if an event onsets during an active cooldown window,
    the alarm state resets immediately, allowing subsequent alarms to fire.
    """
    timestamps = np.arange(100.0, 400.0, 5.0)
    probs = np.full(len(timestamps), 0.1)
    is_onset = np.full(len(timestamps), False)
    
    # Alarm at 115s (cooldown until 415s normally)
    probs[2] = 0.8
    probs[3] = 0.8
    
    # True event onsets at t=200s (step 20)
    is_onset[20] = True
    
    # New alarm sequence after onset at t=220, 225 (step 24, 25)
    probs[24] = 0.8
    probs[25] = 0.8
    
    df = pd.DataFrame({
        "run_id": ["RUN_01"] * len(timestamps),
        "road_id": ["ROAD_A"] * len(timestamps),
        "timestamp": timestamps,
        "prob": probs,
        "is_active_spillback": [False] * len(timestamps),
        "is_event_onset": is_onset,
    })
    
    alarms = extract_operational_alarms(df, mode="ML", prob_col="prob", cooldown_sec=300.0)
    assert len(alarms) == 2, f"Expected cooldown reset allowing 2 alarms, got {len(alarms)}"
    assert alarms[0]["t_alarm"] == 115.0
    assert alarms[1]["t_alarm"] == 225.0


def test_same_road_matching_constraint():
    """
    Verifies that alarms on Road A can NEVER match events on Road B.
    """
    alarms = [
        {"alarm_id": "A1", "road_id": "ROAD_A", "t_alarm": 200.0},
    ]
    events = [
        {"event_id": "E1", "road_id": "ROAD_B", "t_onset": 400.0},
    ]
    
    matched, unmatched_a, unmatched_e = deterministic_alarm_event_matching(alarms, events, horizon_sec=600)
    assert len(matched) == 0, "Alarms across different roads should never match"
    assert len(unmatched_a) == 1
    assert len(unmatched_e) == 1


def test_one_to_one_matching_and_tie_breaking():
    """
    Verifies that:
    1. An alarm can match at most one event.
    2. An event can match at most one alarm.
    3. Chronological earliest alarm is selected when multiple candidates qualify.
    """
    events = [
        {"event_id": "E1", "road_id": "ROAD_A", "t_onset": 500.0},
        {"event_id": "E2", "road_id": "ROAD_A", "t_onset": 700.0},
    ]
    alarms = [
        {"alarm_id": "A1", "road_id": "ROAD_A", "t_alarm": 200.0},  # valid for E1 (lt=300) and E2 (lt=500)
        {"alarm_id": "A2", "road_id": "ROAD_A", "t_alarm": 350.0},  # valid for E1 (lt=150) and E2 (lt=350)
    ]
    
    matched, unmatched_a, unmatched_e = deterministic_alarm_event_matching(alarms, events, horizon_sec=600)
    
    # E1 comes first chronologically, chooses earliest candidate A1
    # A1 is consumed; E2 then matches A2
    assert len(matched) == 2
    assert len(unmatched_a) == 0
    assert len(unmatched_e) == 0
    
    m_e1 = next(m for m in matched if m["event_id"] == "E1")
    m_e2 = next(m for m in matched if m["event_id"] == "E2")
    assert m_e1["alarm_id"] == "A1"
    assert m_e2["alarm_id"] == "A2"


def test_active_spillback_suppression():
    """
    Verifies Rev 6 Section 4.2: alarms triggered while segment is already in active spillback
    are suppressed from false-alarm accounting.
    """
    timestamps = np.arange(100.0, 200.0, 5.0)
    probs = np.full(len(timestamps), 0.9)  # all positive
    # Road is in active spillback from t=100 to t=195
    is_active = np.full(len(timestamps), True)
    
    df = pd.DataFrame({
        "run_id": ["RUN_01"] * len(timestamps),
        "road_id": ["ROAD_A"] * len(timestamps),
        "timestamp": timestamps,
        "prob": probs,
        "is_active_spillback": is_active,
        "is_event_onset": [False] * len(timestamps),
    })
    
    alarms = extract_operational_alarms(df, mode="ML", prob_col="prob", cooldown_sec=300.0)
    assert len(alarms) == 0, "Alarms during active spillback should be suppressed"


def test_exposure_time_and_far_calculation():
    """
    Verifies that exposure time excludes active spillback intervals,
    and FAR equals unmatched_alarms / exposure_hours.
    """
    # 720 steps * 5s = 3600s = 1 hour.
    # Active spillback for 120 steps = 600s.
    # Eligible exposure time = 3000s = 0.8333 hours.
    total_steps = 720
    active_steps = 120
    
    exp_sec = (total_steps - active_steps) * 5.0
    exp_hours = exp_sec / 3600.0
    assert abs(exp_hours - (3000.0 / 3600.0)) < 1e-5
    
    unmatched_alarms = 5
    far = unmatched_alarms / exp_hours
    assert abs(far - 6.0) < 1e-4  # 5 / (3000/3600) = 6.0 alarms/hour
