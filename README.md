# UrbanFlow AI

UrbanFlow AI is a local traffic microsimulator with a live 3D command center. The default classroom network has four connected intersections, 24 directed roads and two lanes per direction. The Python simulator owns vehicle motion and signal permission. The browser renders the positions it receives over a WebSocket.

## Start on Windows

1. Extract the full ZIP to a folder.
2. Install Python 3.10 or newer and enable **Add Python to PATH** during installation.
3. Double-click **START_URBANFLOW.bat**. It reports the exact Python interpreter it will use. If dependencies are missing, it lists them and offers to install them into that interpreter. When dependencies are already available, it skips installation, verifies the Directional v2 checkpoint and four-agent inference, then starts the command center. Keep the launcher window open during the demo.
4. If the browser does not open, visit [the local command center](http://127.0.0.1:8001). Click **Start** to release the first scheduled vehicles.

Three.js and OrbitControls are included, so the dashboard itself does not require a CDN. The first Python dependency installation requires access to a Python package index. On a configured machine, `python run_demo.py --no-browser` starts the same application from a terminal.

The project has no open-source license. Do not redistribute or reuse the source, trained models, or bundled datasets without permission from the project owner and the relevant data providers.

## Classroom demonstration

The dashboard opens on Metro Center with a night map and live telemetry. Use the network picker to switch to Classroom Grid or the other layouts. Search by a street or junction, zoom, switch to an angled 3D view, or fit the full network. The road overlay and road-load chart share four bands based on simulated volume divided by modeled capacity. The playback bar tracks departures due and shows the achieved speed. Analytics use the current run's simulated time.

The route planner estimates travel time from current link speeds and marks the selected path on the map. Use **Incidents** to apply a 90-second road speed restriction and see the countdown. The elapsed time is simulated time. The Metro and Harbor street names and architecture belong to their synthetic networks.

- Select **Classroom grid** and **Rush hour**, then click **Start**. Traffic starts from an empty network. The higher-volume, east–west-heavy part begins at 19.5 simulated seconds.
- At **1x**, watch vehicles approach the stop bars and queue while their signals are red. Use **4x** or **20x** to move through the run faster. Actual speed depends on network size and the selected intelligence modes. The simulator uses the same 50 ms physics step at every playback speed.
- Choose a junction from **Intersection Status** for a close view. Scroll to zoom at the pointer and double-click to center a map location. Drag to pan in Top-Down 2D, or drag to orbit and right-drag to pan in 3D. **Inspect Vehicle** selects an active vehicle. Use **Follow Vehicle** to follow it. The **Playback** toggle hides or restores the controls. Its clock shows simulated minutes and seconds. When the full demand finishes, **Replay** starts the same schedule again. The live fleet card counts vehicles currently moving.
- Start with Fixed-Time, then select Adaptive and reset to watch the same schedule under queue- and arrival-responsive control. **Reset Scenario** also replays the currently selected schedule from the beginning.
- Open **Traffic**, then **Open comparison** to compare Fixed-Time, Actuated, MaxPressure and CooperativePredictive using the selected network and demand, including uploads. The four runs start empty with routing off. The window is at least four minutes and extends 120 seconds beyond the last departure up to a 30-minute cutoff. The result ranks completed trips first, average queue second and completed-trip waiting third. Apply a recommendation directly to the live run.

- Under **Routing and overlays** in **Controls**, select **Compare routing modes** to replay the same network and demand with **Off**, **Reactive**, and **Predictive** while holding the selected signal controller fixed. Off keeps the original paths. Reactive considers queues and road space right now. Predictive also estimates where queues may spill back. The comparison shows accepted reroutes alongside trip and queue outcomes.

## City networks and RL evidence

Midtown Avenues is a compact synthetic city grid. The six selectable networks are Classroom Grid, Midtown Avenues, Metro Center, Harbor City, Bronx, NY, and Single Corridor. The synthetic grids share the classroom simulator’s signal, lane, and turn-clearance rules.

**Metro center** has 36 signalized junctions and 168 directed links. **Harbor city** has 48 signalized junctions and 220 directed links. Both synthetic layouts have two lanes per direction, cross-town arterials, parks and distinct building heights. Use **Traffic → Focus a junction** to jump to any signal. **Commuter reversal** shifts the dominant entry side mid-run. **Event release** sends a short pulse followed by recovery. Both are synthetic stress tests.

Under **Traffic demand**, the **NYC DOT measured counts** group contains three one-hour windows from Bronx segments near W Farms Road and East 172nd Street. The bundled [raw count subset](data/traffic_counts/nyc-dot-bronx-2013.csv) preserves each 15-minute count by direction. UrbanFlow generates timestamps, routes and lane choices from those counts. The official [NYC DOT Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt) page describes the data collection.

Under **Traffic**, open **RL benchmark results** for the three new measured-count replays and nine archived SUMO studies. The dashboard ranks mean time loss plus departure delay across shared seeds and links to the original CSVs. The same saved `ppo_real_mix_s0` checkpoint led the January 10 evening count replay. SUMO Actuated led the January 8 morning and January 9 midday replays. RL entries lead six archived studies. The live UrbanFlow comparison does not execute these PPO, IPPO or MAPPO checkpoints.

Reproduce the new comparisons with `python scripts/benchmark_observed_demand.py PATH_TO_MARL_TRAFFIC PATH_FOR_NEW_RESULTS` from an environment with the research project's SUMO and Stable-Baselines3 dependencies. The test fixes the checkpoint, the three windows and seeds 4101 through 4105. Each controller receives the same pre-generated vehicles, times and turning routes for a seed. Turn proportions are assumptions. The benchmark uses the compatible single SUMO intersection.

The archived single-intersection runs can be reproduced in a separate copy of the research repository with `python scripts/eval_scenarios_baselines_test.py`. The evaluator uses test seeds 3001 through 3005 and its original checkpoint and SUMO network configuration. It writes the replay CSVs into `results/final_test` in the copy.

## Saved and live comparisons

**Traffic → Open comparison** immediately shows the saved SUMO benchmark for any of the eight built-in demand presets. This is a single-intersection reference and always uses original traffic volume. The benchmark table keeps the measured ranking and highlights the RL entry. Use **RL benchmark results** to browse earlier studies and download the per-seed scores. To reproduce the five synthetic reference benchmarks, add `--presets` to the benchmark command above.

Upload a schedule or NYC DOT counts CSV to enable the live controller and routing comparisons on the selected map. The replay includes the uploaded schedule's final departure and a clearance window. The live city comparison uses UrbanFlow's built-in controllers. The pretrained PPO results come from SUMO.

**Traffic volume** increases scheduled vehicles up to 4× and restarts the current run. Returning to 1× restores the original schedule. At high playback settings the clock shows achieved speed, which depends on how many vehicles and junctions are active.

## Demand files

Select the intended network before uploading. **Download re-uploadable schedule** exports the current demand with its exact map routes. The uploader also accepts the bundled raw NYC DOT count CSV. It places recorded hours consecutively, preserving directional and quarter-hour totals. Controller score CSVs are comparison results. Departure schedule CSVs need a header row. JSON files contain an array of objects with the same field names. The file must be UTF-8 and at most 2 MB. The validator checks up to 10,000 vehicle records before applying a new schedule. A successful upload resets and pauses the simulation.

| Field | Meaning |
|---|---|
| `vehicle_id` | Unique vehicle identifier |
| `departure_time` | Seconds from the start |
| `entry_road` | Directed boundary entry road ID |
| `entry_lane` | Optional zero-based lane index, default 0 |
| `destination` | Directed boundary exit road ID |
| `vehicle_type` | Optional: sedan, van, bus or emergency. Default sedan |
| `desired_speed` | Optional desired speed in m/s. Defaults to the vehicle type's speed and is bounded by the active road limit in the simulator |
| `route` | Optional JSON array of connected road IDs. The schedule export includes it to preserve the original route |

Use [classroom-example.csv](data/classroom-example.csv) as a template. The [topology endpoint](http://127.0.0.1:8001/api/topology) lists the valid road IDs. Classroom entry road IDs contain `_in_`; exit road IDs contain `_out_`. Built-in Normal, Rush Hour, Peak Demand, Lane Closure, Road Incident and Downstream Gridlock are synthetic scenarios. Restrictions activate at 20 simulated seconds.

## How to read the results

- **Completed** counts vehicles that reached their destination. **Incomplete** counts entered vehicles still on the network at the cutoff. **Never entered** counts due vehicles still waiting to enter at that cutoff. Departures scheduled for after the cutoff are tracked separately.
- The comparison's speed is the mean of active vehicle speed observations sampled each second. Travel time and wait are measured among completed trips. Throughput expresses actual exits over the run as an hourly rate.
- Queue counts sample stopped approach vehicles each second. Spillback seconds count samples with a stopped vehicle within 10 m of the upstream end of an internal road. The live congestion index is a speed-based indicator.
- Classroom signals use NS/EW phases, with explicit left, straight and right connections. Parallel straight lanes and separated right turns can clear together. Left turns and overlapping paths use exclusive clearance. Vehicles reserve enough downstream space to clear the junction with their entire body. Green is bounded to 10–60 seconds, with three seconds of yellow and at least 1.5 seconds of all-red. All-red can extend while a committed vehicle clears.
- The comparison describes this synthetic network and schedule. Model-based forecasts and predicted routing cost changes are estimates. Bronx and Corridor remain available as separate environments. Both comparisons and demand uploads are available on Classroom Grid, Single Corridor and Bronx. The Bronx simulation models one driving lane per direction, reserves irregular junctions exclusively, and uses sedan-length traffic. Its built-in demand favors links with room to clear junctions.

## Source and research

`urbanflow/core` contains the road network, demand processing and simulator. `urbanflow/server` runs the control and telemetry pipeline. `urbanflow/web` contains the dashboard. `urbanflow/intelligence` contains the existing controllers, forecasting, capacity and routing work.

The supplied model checkpoints, forecast datasets, and historical result files are included. Older experimental checkpoints remain archived for reproducibility; they do not add selectable networks to the dashboard. See [Research README](docs/RESEARCH_README.md), [Architecture](docs/ARCHITECTURE.md) and [Classroom comparison output](docs/classroom-comparison.json). To run the tests from an extracted source directory, install the project with `python -m pip install -e ".[dev]"`, then run `python -m pytest -q`.

## Classroom Grid on Windows

Open Command Prompt in this project folder with Python 3.10 or 3.11 on PATH. If you use a virtual environment, activate it before launching.

```bat
START_URBANFLOW.bat
```

The launcher prints the Python executable it uses for dependency checks, installation, inference, and the server. If packages are missing, type Y to install them, or run `INSTALL_RL_DEPENDENCIES.bat` to install and verify without starting the server. Use `START_URBANFLOW.bat --check-only` to check an existing environment without installing packages. The launcher verifies Directional v2 on CPU before starting and reports four loaded agents. The dashboard opens paused with Rush Hour at 1×, routing off, and the trained **Multi-Agent RL Model** selected. Press **Start** and watch the four agents in **Multi-Agent RL — Agent Status**. Use **Comparison** to open the selected archived comparison and expand **Research Results** to see the complete benchmark. **Experiments** runs matched controller evaluations. The recent decisions table reports what each agent requested and the signal state at the time of that request.

See [Classroom Grid policy diagnostics](docs/diagnostics/REPORT.md) for how Directional v2 was selected and performance on held-out directional traffic.

**Route Test Drive** opens as a disclosure beneath Route planner. Opening it creates a route preview. **Drive and Compare** runs an isolated comparison on the selected traffic schedule. Expand **Test Details** to see the exact checkpoint, experiment conditions, and per-intersection waits or download the decision log. See the [route investigation](docs/diagnostics/ROUTE_TEST_REPORT.md) and recorded repeated experiments.
