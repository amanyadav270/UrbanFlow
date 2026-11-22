# UrbanFlow AI — Intelligent Urban Traffic Management Platform & 3D Digital Twin

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)](#prerequisites)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-green)](#prerequisites)
[![Tests](https://img.shields.io/badge/Tests-191%20Passing-brightgreen)](#testing--verification)
[![License](https://img.shields.io/badge/License-MIT-purple)](#license)

UrbanFlow AI is an end-to-end, high-performance urban mobility platform combining a **real-time 3D geographic digital twin**, **multi-agent reinforcement learning (MARL) signal control**, **multi-horizon supervised congestion forecasting**, and **predictive capacity-aware dynamic routing**.

Built on a real-world geographic corridor network (Bronx Fordham Road & Grand Concourse, NYC), UrbanFlow models microscopic vehicle interactions, simulates traffic signal controllers with millisecond precision, predicts arterial spillback before queues materialize, and dynamically reroutes vehicles around bottlenecks while strictly preserving downstream corridor capacities.

---

## 1. Executive Summary & Capabilities

- **High-Fidelity 3D Urban Digital Twin:** Rendered in WebGL via Three.js with extruded asphalt roadbeds, multi-lane dual carriageways, road divider medians, zebra crossings, signal gantries with illuminated phase bulbs (Green / Yellow / Red), procedural cityscape buildings, and daytime/nighttime atmospheric lighting.
- **Microscopic Traffic Simulation Engine:** Dual kinematic IDM (Intelligent Driver Model) car-following and MOBIL lane-changing engine supporting multi-class fleets (sedans, delivery vans, transit buses, emergency vehicles). Fully interchangeable with microscopic SUMO via the abstract `SimulatorAdapter` contract.
- **Multi-Horizon Supervised Intelligence Pipeline:** Native LightGBM booster inference predicting average speed, density, queue lengths, and congestion probabilities at 5-minute, 10-minute, and 15-minute horizons, fed by a sliding 28-feature rolling observation buffer.
- **Calibrated Spillback Risk Classification:** 10-minute calibrated network spillback classifier detecting incipient corridor gridlock before physical queue spillover occurs.
- **Capacity-Aware Dynamic Routing with 10 Safeguards:** Dynamic route evaluation incorporating operational capacity, V/C ratios, and predictive spillback probabilities. Enforces strict safeguards: 25% diversion quota, 180s vehicle cooldown, 15s/10% minimum travel-time improvement, 0.85 V/C admission cap, 0.80 physical storage limit, and counterfactual reservation tracking to prevent creating secondary bottlenecks.
- **Four Coordinated Signal Control Regimes:**
  1. `Fixed-Time`: Deterministic cyclic split coordination.
  2. `Actuated`: Microscopic gap-out, minimum green extension, and vehicle actuation.
  3. `MaxPressure`: Upstream vs. downstream queue differential pressure balancing with storage safeguards.
  4. `CooperativePredictive`: Multi-agent platoon anticipation, green waves, downstream storage veto, and 10-minute predictive spillback risk mitigation.
- **Operations HUD & Command Center:** 20 Hz WebSocket streaming telemetry, live KPI cards, interactive vehicle fleet tracking with route visualization, manual incident injection, peak rush-hour surge triggering, and instant scenario loading.

---

## 2. One-Click Quick Start (Windows)

UrbanFlow AI is engineered for seamless one-click execution on Windows 10/11:

### Method A — Windows Batch Launcher (Recommended)
Double-click `START_URBANFLOW.bat` in the project root, or launch from PowerShell / Command Prompt:

```cmd
START_URBANFLOW.bat
```

The launcher automatically:
1. Detects your local Python 3.10+ installation.
2. Sets `PYTHONPATH` to include the project root and packages.
3. Verifies all required dependencies (`fastapi`, `uvicorn`, `lightgbm`, `joblib`, `numpy`).
4. Checks for port collisions on the configured port (8001 by default).
5. Starts Uvicorn and opens the dashboard at `http://127.0.0.1:8001` by default.

### Method B — PowerShell Launcher
```powershell
.\START_URBANFLOW.ps1
```
*Optional parameters:*
- `.\START_URBANFLOW.ps1 -Restart`: Replaces a verified UrbanFlow listener on the configured port.
- `.\START_URBANFLOW.ps1 -NoBrowser`: Starts the backend server without launching the browser window.

### Method C — Standard Python CLI
```powershell
$env:PYTHONPATH = ".;urbanflow"
python run_demo.py --open
```

Once running, navigate to:
- **Command Center UI:** [http://127.0.0.1:8001](http://127.0.0.1:8001)
- **API Health Status:** [http://127.0.0.1:8001/api/status](http://127.0.0.1:8001/api/status)
- **Live Telemetry WebSocket:** `ws://127.0.0.1:8001/ws/live`

---

## 3. Four-Layer System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       LAYER 4: OPERATIONS & UI                          │
│   Three.js 3D WebGL Digital Twin  │  Command Center HUD  │  WebSockets  │
│   - Multi-camera views (Orbit, Tactical, Signal Zoom, Follow Chase Cam) │
│   - Real-time 3D vehicle trails, signal bulbs, and incident markers     │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │ 20 Hz State Broadcast
┌────────────────────────────────────┴────────────────────────────────────┐
│                    LAYER 3: INTELLIGENCE & CONTROL                      │
│  Traffic State Estimator     │ Road Capacity & LOS (Highway Capacity Manual)
│  Modular Congestion Predictor│ 28-Feature Online Observation Buffer     │
│  Spillback Classifier (10m)  │ Predictive Capacity-Aware Routing Engine │
│  Signal Controllers: Fixed-Time | Actuated | MaxPressure | Cooperative  │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │ State / Feedback
┌────────────────────────────────────┴────────────────────────────────────┐
│                  LAYER 2: SIMULATOR ADAPTER & INTERFACE                 │
│                     SimulatorAdapter Abstract Contract                  │
│       - apply_signal_phases()    - reroute_vehicles()                   │
│       - inject_incident()        - trigger_congestion_surge()           │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │                                                   │
┌──────────┴─────────────────────────┐   ┌─────────────────────┴──────────┐
│      LAYER 1A: KINEMATIC SIM       │   │       LAYER 1B: SUMO CORE      │
│ - IDM car-following (20 Hz)        │   │ - Eclipse SUMO Micro-Simulator │
│ - MOBIL lane selection             │   │ - TraCI / libsumo bindings     │
│ - Geometric intersection FSM       │   │ - High-fidelity sub-lane model │
└────────────────────────────────────┘   └────────────────────────────────┘
```

---

## 4. Key Features & How to Demonstrate

### 1. Preset Demonstration Scenarios
Select any scenario directly from the **Demonstration Scenarios** control card on the dashboard or run via the API:
- `🟢 Normal Flow`: Resets network to balanced baseline demand. Roads operate at Level of Service A/B with free-flow speeds (~30 km/h).
- `🟡 Peak Surge`: Triggers an immediate rush-hour influx of 30+ vehicles into major arterial corridors, elevating density and testing signal adaptation.
- `🔴 Incident Detour`: Places an active road incident and lane restriction on the Grand Concourse corridor (severity 0.8, 90s duration). Watch the capacity estimator drop available capacity and the dynamic routing engine divert approaching traffic onto parallel avenues.
- `🟣 Gridlock Veto`: Injects extreme downstream saturation to demonstrate the 10 safeguards in action. The spillback classifier and storage safeguard veto diversions that would clog secondary streets, preventing cascading gridlock.

### 2. Four Signal Control Regimes
Switch between controllers using the dropdown in the **Traffic Signal Regime** panel:
- **Fixed-Time:** Classic pre-timed cycles (North-South / East-West phase splits). Baseline control.
- **Actuated:** Dynamic green extension triggered by approaching vehicle detectors, terminating phases when headways exceed passage thresholds.
- **MaxPressure:** Evaluates upstream queue accumulation minus downstream available storage for each movement, actively equalizing network pressure.
- **CooperativePredictive:** Coordinates multi-intersection green waves, checks downstream storage limits to prevent spillback gridlock, and integrates 10-minute ML congestion predictions.

### 3. Dynamic Routing Modes
Control rerouting behavior in the **Dynamic Routing & Diversion** panel:
- **OFF:** Vehicles adhere strictly to scheduled baseline routes. Zero diversions.
- **REACTIVE:** Responds dynamically to real-time $V/C$ ratios and operational capacity drops.
- **PREDICTIVE:** Ingests the 28-feature rolling network state and evaluates the calibrated LightGBM classifier to divert traffic before queue spillback forms.

### 4. Interactive Vehicle Inspection
Click `🔍 Track Vehicle` on the **Active Vehicle Fleet** card or click any 3D vehicle in the viewport. The camera automatically tracks the vehicle, displaying its current speed, route index, remaining waypoints, reroute count, and visual trajectory trail.

---

## 5. Automated Scenario Benchmark Demonstration

To execute a comprehensive, deterministic terminal demonstration across all scenarios, routing modes, and controllers, run:

```powershell
python scripts/demonstrate_scenarios.py
```

This runs a rapid, fully headless multi-condition evaluation and outputs clean performance metrics:
1. Operational scenarios response (Active vehicles, completion count, mean velocity, congestion index).
2. Routing comparison (`OFF` vs. `REACTIVE` vs. `PREDICTIVE`) under incident conditions.
3. Signal controllers comparison across 27 signalized intersections.

---

## 6. Research Findings, Benchmark Results & Honest Limitations

### Multi-Seed Benchmark Summary (Phase 6E)
In rigorous 20-seed evaluations on the calibrated Bronx corridor:
- **Actuated Control** demonstrated the highest raw post-warmup throughput (3,700.7 vehicles completed, 3,830.3 veh/hr) under steady-state demand due to zero phase overhead during low-volume intervals.
- **CooperativePredictive Control** demonstrated superior queue stabilization and spillback prevention during peak-surge regimes, preventing upstream gridlock through downstream storage vetoes.
- **MaxPressure** proved highly effective at corridor pressure equalization, though vulnerable when downstream exit links lack storage capacity.

### Fair Demand-Based System Cost Audit
Independent counterfactual audit of all scheduled vehicles revealed:
- Standard throughput metrics can exhibit survivorship bias by omitting vehicles queued outside the network entry gates.
- Including unentered vehicle delay emphasizes the necessity of entry-capacity coordination: aggressive signal clearing on internal corridors without entry metering can shift bottlenecks to boundary links.

### System Limitations
- **SUMO Adapter Fallback:** While the native kinematic engine runs at 100x real-time without dependencies, the optional SUMO adapter requires a local `sumo` installation and `libsumo` or `traci` Python packages.
- **LightGBM Native Booster Dependency:** In Windows environments with Python 3.11, scikit-learn wrapper unpickling can encounter C-extension linkage issues; UrbanFlow solves this by loading native LightGBM `Booster` representations directly.

---

## 7. Testing & Verification

UrbanFlow AI features an exhaustive test suite covering all modules:

```powershell
pytest tests -v
```

**Results:** Exactly **191 / 191 tests passing** (0 errors, 0 failures, 0 skipped) across:
- `test_kinematic_sim.py`: Kinematic physics, IDM car following, and MOBIL lane changes.
- `test_intelligence.py`: State estimation, capacity calculation, and LOS classification.
- `test_classical_signal_controllers.py`: Fixed-Time, Actuated, and MaxPressure logic.
- `test_cooperative_predictive_signal_controller.py`: Cooperative predictive coordination and green waves.
- `test_spillback_classifier_pipeline.py`: Spillback classifier feature extraction and probability calibration.
- `test_phase8_1_predictor_stability.py`: Native LightGBM booster stability and 28-feature rolling observation buffer.
- `test_phase8_2_dynamic_routing_live.py`: Real-time dynamic routing, 10 safeguards, and live WebSocket telemetry.
- `test_server.py`: ASGI REST endpoints, WebSocket streaming, and client control actions.

---

## 8. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Port 8001 in use** (`[Errno 10048]`) | Another process is listening. | Use `python run_demo.py --restart` for an existing UrbanFlow instance or set `URBANFLOW_PORT` to a free port. |
| **Blank 3D Canvas** | WebGL hardware acceleration disabled in browser. | In Chrome/Edge, navigate to `chrome://settings/system` and enable "Use graphics acceleration when available". |
| **Missing Model Warning** | Pretrained `.joblib` or Booster files not found in `data/models`. | UrbanFlow automatically falls back to baseline operational capacity heuristics without crashing. |

---

## 9. License

This project is licensed under the MIT License — see the LICENSE file for details.
