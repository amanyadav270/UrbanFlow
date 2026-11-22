# UrbanFlow AI — Evaluator & Demonstration Guide

This guide provides step-by-step instructions for launching, exploring, and evaluating the **UrbanFlow AI** Digital Twin and Intelligent Traffic Operations platform.

---

## 1. Fast Launch Instructions

### Recommended Launch (Windows)
Double-click **`START_URBANFLOW.bat`** in the project directory:
```cmd
START_URBANFLOW.bat
```
*Or via PowerShell:*
```powershell
.\START_URBANFLOW.ps1
```

### Direct CLI Launch
```powershell
cd D:\TRAFFIC\urbanflow
$env:PYTHONPATH = ".;urbanflow"
python run_demo.py --open
```

- **Browser Dashboard URL:** [http://127.0.0.1:8001](http://127.0.0.1:8001)
- **API Status Endpoint:** [http://127.0.0.1:8001/api/status](http://127.0.0.1:8001/api/status)
- **Live WebSocket Feed:** `ws://127.0.0.1:8001/ws/live`

Use `--restart` to replace a verified UrbanFlow listener on the configured port.

---

## 2. Command Center Walkthrough

When the dashboard opens, you will see a dark glassmorphic Command Center:
- **Top Navigation Bar:** Live status pill (`ONLINE 20 Hz`), current simulation clock, active signal regime badge, and dynamic routing status pill.
- **Top KPI Cards:** Real-time metrics streaming at 20 Hz:
  - *Active Vehicles* & *Completed Trips*
  - *Network Throughput* (veh/hr)
  - *Average Fleet Velocity* (km/h)
  - *Network Congestion Index* ($0.0 \to 1.0$)
- **Central 3D Viewport:** Three.js WebGL digital twin of the Bronx Fordham Road / Grand Concourse corridor with moving vehicles, signal gantries with illuminated phase lights, crosswalks, and 3D buildings.
- **Left Control Panels:**
  - *Demonstration Scenarios* (One-click presets: Normal, Peak Surge, Incident Detour, Gridlock Veto).
  - *Camera Controls* (Strategic 3D Orbit, Tactical 2D Top-Down, Signal Intersection Zoom, Follow-Vehicle Chase Cam).
  - *Visual Environment* (Day / Night lighting toggle).
  - *Active Vehicle Fleet* (Live vehicle count, reroute tallies, and `🔍 Track Vehicle` button).
- **Right Analytics Panels:**
  - *Traffic Signal Regime* (Dropdown to switch between Fixed-Time, Actuated, MaxPressure, and CooperativePredictive).
  - *Dynamic Routing & Diversion* (Mode switcher: OFF, REACTIVE, PREDICTIVE, with live divert counters and safeguard telemetry).
  - *Network Bottlenecks & Multi-Horizon AI* (Real-time speed, density, queue length, and 10-minute ML spillback probabilities).

---

## 3. Step-by-Step Scenario Demonstrations

### Scenario 1: Normal Traffic Flow (Baseline)
1. Click the **`🟢 Normal Flow`** button in the Demonstration Scenarios card.
2. **What to Observe:**
   - Balanced vehicle arrivals across east-west and north-south approaches.
   - Low congestion index ($\approx 0.35 - 0.42$).
   - Average speed steady at $\sim 25 - 30\text{ km/h}$.
   - All roads in the bottleneck panel show Level of Service A or B (green).

### Scenario 2: Peak-Hour Congestion Surge
1. Click the **`🟡 Peak Surge`** button.
2. **What to Observe:**
   - 30 additional vehicles are instantly injected into the major inbound arterials.
   - Active vehicle count jumps from ~45 to 75+.
   - Queues start forming at key intersections.
   - The signal controller adapts to prioritize dense platoons, extending green phases.

### Scenario 3: Road Incident & Dynamic Diversion
1. In the **Dynamic Routing & Diversion** panel, select **`PREDICTIVE`** mode.
2. Click the **`🔴 Incident Detour`** button.
3. **What to Observe:**
   - A red visual incident marker appears on the Grand Concourse corridor.
   - Available capacity drops by 80% on the affected segment.
   - The 28-feature rolling observation buffer flags elevated spillback risk.
   - The routing engine identifies vehicles approaching the incident and calculates alternative routes.
   - Notice the **Accepted Diverts** counter increment on the dashboard.
   - Diverted vehicles dynamically change direction onto parallel avenues, maintaining corridor flow.

### Scenario 4: Downstream Gridlock & Safeguard Vetoes
1. Click the **`🟣 Gridlock Veto`** button.
2. **What to Observe:**
   - Heavy saturation and a severe blockage are injected downstream.
   - Notice that while vehicles request detours, the **Rejected Diverts** counter increases.
   - Inspect the **Rejection Reasons** breakdown: safeguards like `Storage Limit Exceeded (80%)` and `Spillback Risk Exceeded (>0.60)` prevent dumping traffic into already saturated side streets.
   - This demonstrates the safety architecture preventing secondary gridlock.

---

## 4. Signal Controller Comparison

You can switch signal controllers in real time during any scenario:

| Controller | How to Select | Key Behavior to Verify |
| :--- | :--- | :--- |
| **Fixed-Time** | Select *Fixed-Time* in dropdown | Pre-timed, regular cyclic splits. Does not respond to queue buildup; queues will accumulate during surges. |
| **Actuated** | Select *Actuated* in dropdown | Vehicle detectors gap out empty approaches and hold green for dense platoons up to max-green limit (60s). High throughput under steady flow. |
| **MaxPressure** | Select *MaxPressure* in dropdown | Computes queue differentials between upstream approach and downstream exit links. Prevents sending cars into blocked downstream links. |
| **CooperativePredictive** | Select *CooperativePredictive* in dropdown | Coordinates adjacent intersections, provides green waves along arterial corridors, and integrates 10-minute ML spillback probabilities. |

---

## 5. Vehicle Inspection & Tracking

1. Click **`🔍 Track Vehicle`** on the Active Vehicle Fleet card (or click any 3D car in the viewport).
2. The camera switches to **Follow Vehicle Chase Cam**, smoothly following the car through the network.
3. An inspector popover appears showing:
   - Vehicle ID (e.g., `veh_0042`) and vehicle type (Sedan / Van / Bus / Emergency).
   - Current velocity and acceleration.
   - Current road ID and route progression index.
   - Total times rerouted by dynamic routing.
   - Full list of remaining waypoints on its journey.
4. Click **`Reset Camera`** to return to standard 3D Orbit view.

---

## 6. Automated Terminal Demonstration

For a fast, reproducible command-line verification without the browser:

```powershell
python scripts/demonstrate_scenarios.py
```

This headless script runs all 4 scenarios, 3 routing modes, and 4 signal controllers in under 50 seconds and outputs clean comparison tables.

---

## 7. Verification Test Suite

To run the complete automated test suite (191 tests):

```powershell
pytest tests -v
```

Expected result: `191 passed in ~10s`.
