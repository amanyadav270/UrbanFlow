/**
 * UrbanFlow AI — Operations Command Center HUD Controller
 * Manages DOM interaction, real-time telemetry card updates, and control events.
 */

const marlNode = id => document.getElementById(id);
const MARL = "PPO_NEIGHBOR";
const AGENTS = [["INT_NW", "North-West"], ["INT_NE", "North-East"], ["INT_SW", "South-West"], ["INT_SE", "South-East"]];

export class HUDController {
  constructor(app) {
    this.app = app;

    this._bindElements();
    this._setupEventListeners();
  }

  _bindElements() {
    this.elThroughput = document.getElementById("val-throughput");
    this.elAvgSpeed = document.getElementById("val-speed");
    this.elAvgDelay = document.getElementById("val-delay");
    this.elActiveVehicles = document.getElementById("val-vehicles");
    this.elArrived = document.getElementById("val-arrived");

    // Signal HUD
    this.elSignalName = document.getElementById("signal-phase-name");
    this.elSignalTimer = document.getElementById("signal-phase-timer");
    this.dotRed = document.getElementById("light-red");
    this.dotYellow = document.getElementById("light-yellow");
    this.dotGreen = document.getElementById("light-green");

    // Comparison Card

    // Traffic Intelligence Elements
    this.isHeatmapOn = false;
    this.currentHorizon = "10min";
    this.elBottleneckList = document.getElementById("bottleneck-list");
    this.elBottleneckBadge = document.getElementById("badge-bottleneck-count");
    this.elHorizonTag = document.getElementById("intel-horizon-tag");

    // Signal Controller HUD Elements
    this.elCtrlActiveName = document.getElementById("ctrl-active-name");
    this.elCtrlModelTag = document.getElementById("ctrl-model-tag");
    this.elHudCtrlName = document.getElementById("hud-ctrl-name");
    this.elHudCtrlSignals = document.getElementById("hud-ctrl-signals");

    // Dynamic Routing HUD Elements
    this.elRoutingActiveName = document.getElementById("routing-active-name");
    this.elRoutingModelTag = document.getElementById("routing-model-tag");
    this.elValRerouteCount = document.getElementById("val-reroute-count");
    this.elValRerouteRejected = document.getElementById("val-reroute-rejected");
    this.elValReservationsCount = document.getElementById("val-reservations-count");
    this.elBadgeRerouteStatus = document.getElementById("badge-reroute-status");
    this.elRerouteDecisionsList = document.getElementById("reroute-decisions-list");
    this.elHudRoutingMode = document.getElementById("hud-routing-mode");

    // Vehicle Inspector
    this.inspector = document.getElementById("vehicle-inspector");
    this.inspId = document.getElementById("insp-id");
    this.inspType = document.getElementById("insp-type");
    this.inspSpeed = document.getElementById("insp-speed");
    this.inspRoad = document.getElementById("insp-road");
    this.inspRouteProgress = document.getElementById("insp-route-progress");
    this.inspRerouteCount = document.getElementById("insp-reroute-count");
    this.inspRouteGuidance = document.getElementById("insp-route-guidance");
    this.inspWait = document.getElementById("insp-wait");
  }

  _setupEventListeners() {
    document.querySelectorAll("[data-panel]").forEach(button => {
      const panel = document.getElementById(button.dataset.panel);
      button.addEventListener("click", () => {
        panel.hidden = !panel.hidden;
        button.setAttribute("aria-expanded", String(!panel.hidden));
      });
    });

    document.getElementById("open-rl-benchmarks").addEventListener("click", () => this.openRlBenchmarks());
    document.getElementById("junction-select").addEventListener("change", event => {
      if (event.target.value) this.app.focusIntersection(event.target.value);
    });

    const mapSelect = document.getElementById("map-select");
    mapSelect.addEventListener("change", () => {
      this.app.sendCommand({ action: "load_map", map_name: mapSelect.value }, `${mapSelect.selectedOptions[0].textContent} loaded`);
    });
    const scenarioSelect = document.getElementById("scenario-select");
    scenarioSelect.addEventListener("change", () => {
      this.app.sendCommand({ action: "load_scenario", scenario: scenarioSelect.value },
        `${scenarioSelect.selectedOptions[0].textContent} loaded. Run restarted.`);
    });
    const trafficFactor = document.getElementById("traffic-factor");
    trafficFactor.addEventListener("change", () => {
      this.app.sendCommand({ action: "set_traffic_factor", factor: Number(trafficFactor.value) }, "Traffic volume changed. Run restarted.");
    });

    const upload = document.getElementById("demand-upload");
    upload.addEventListener("change", async () => {
      const file = upload.files[0];
      if (!file) return;
      const status = document.getElementById("demand-error");
      status.hidden = true;
      try {
        const format = file.name.split(".").pop().toLowerCase();
        const response = await fetch(`/api/demand?format=${encodeURIComponent(format)}`, {
          method: "POST",
          body: file,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.message);
        this.updateScenarioSelection("uploaded");
        this.setPlayback(true);
        this.updateDemand(result, result.demand_revision);
        this.showFeedback(`Demand loaded: ${result.scheduled} scheduled vehicles. Press Start to run.`);
      } catch (error) {
        status.textContent = error.message;
        status.hidden = false;
      } finally {
        upload.value = "";
      }
    });
    document.getElementById("compare-classroom").addEventListener("click", () => this.openStudy("controller"));
    document.getElementById("compare-routing").addEventListener("click", () => this.openStudy("routing"));
    document.getElementById("study-run").addEventListener("click", () => this.runComparison());
    document.getElementById("held-out-results").addEventListener("toggle", event => {
      if (event.currentTarget.open) this.loadHeldOut();
    });

    marlNode("activate-marl").addEventListener("click", async () => {
      this.modelApplying = true;
      this.updateClassroomController();
      await this.app.sendCommand({action: "set_controller", controller: MARL},
        "Trained model loaded · 4 agents ready. Press Start.");
      this.modelApplying = false;
      this.updateClassroomController();
    });

    // 3. Traffic Signal Controller Selection
    document.querySelectorAll("[data-controller]").forEach(btn => {
      btn.addEventListener("click", () => {
        const ctrlId = btn.getAttribute("data-controller");
        this.app.sendCommand({ action: "set_controller", controller: ctrlId },
          `${btn.textContent.trim()} selected for upcoming signal decisions.`);
      });
    });

    // 4. Dynamic Routing Mode Selection
    document.querySelectorAll("[data-routing]").forEach(btn => {
      btn.addEventListener("click", () => {
        const mode = btn.getAttribute("data-routing");
        this.app.sendCommand({ action: "set_routing_mode", mode }, `Routing set to ${btn.textContent.trim()}.`);
      });
    });

    document.querySelectorAll("[data-cam]").forEach(button => {
      button.addEventListener("click", () => {
        const mode = button.dataset.cam;
        this.app.dashboard.setTestCarCamera(false);
        if (mode === "chase") {
          const selected = this.app.vehicleManager.getSelectedVehicle() || this.inspectVehicle();
          if (!selected) return;
          this.app.scene3d.setCameraView(mode, selected.mesh);
        } else {
          this.app.scene3d.setCameraView(mode);
        }
        this.updateCameraSelection();
      });
    });

    // 7. Heatmap Overlay Toggle
    const heatBtn = document.getElementById("btn-toggle-heatmap");
    if (heatBtn) {
      heatBtn.addEventListener("click", () => {
        this.isHeatmapOn = !this.isHeatmapOn;
        heatBtn.classList.toggle("active", this.isHeatmapOn);
        heatBtn.setAttribute("aria-pressed", String(this.isHeatmapOn));
        if (this.app.roadBuilder) {
          this.app.roadBuilder.setHeatmapMode(this.isHeatmapOn);
        }
      });
    }

    const playBtn = document.getElementById("btn-play-pause");
    playBtn.addEventListener("click", () => {
      this.app.sendCommand({ action: playBtn.dataset.paused === "true" ? "resume" : "pause" });
    });
    const speed = document.getElementById("playback-speed");
    speed.addEventListener("change", () => {
      this.app.sendCommand({ action: "set_speed", multiplier: Number(speed.value) },
        `Playback speed set to ${speed.selectedOptions[0].textContent}.`);
    });
    document.getElementById("btn-reset-sim").addEventListener("click", () => {
      this.app.dashboard.setPlannedRouteHighlight(false);
      this.app.sendCommand({ action: "reset" }, "Run reset to the beginning of the selected demand.");
    });

    document.getElementById("btn-inspect-fleet").addEventListener("click", () => this.inspectVehicle());

    // Inspector Close
    const closeInsp = document.getElementById("close-inspector");
    if (closeInsp) {
      closeInsp.addEventListener("click", () => {
        if (this.inspector) this.inspector.style.display = "none";
        this.app.vehicleManager.selectedVehicleId = null;
        if (this.app.scene3d.cameraMode === "chase") {
          this.app.scene3d.setCameraView("strategic");
          this.updateCameraSelection();
        }
        this.app.dashboard?.setPlannedRouteHighlight(this.app.dashboard.showPlannedRoute);
      });
    }
  }

  updateMapSelection(mapName) {
    document.getElementById("map-select").value = mapName;
    this.setConnection(this.app.isConnected);
    this.refreshLearnedControllers(mapName);
    this.updateClassroomController();

  }

  async refreshLearnedControllers(mapName) {
    const buttons = document.querySelectorAll('[data-controller^="PPO_"]');
    const request = this.learnedRequest = (this.learnedRequest || 0) + 1;
    this.learnedReason = undefined;
    for (const button of buttons) button.disabled = true;
    try {
      const response = await fetch("/api/learned-controllers");
      if (!response.ok) throw new Error("Could not check checkpoint availability.");
      const data = await response.json();
      if (request !== this.learnedRequest || data.map_name !== mapName || this.app.mapName !== mapName) return;
      this.learnedReason = data.choices[MARL];
      for (const button of buttons) {
        const reason = data.choices[button.dataset.controller];
        button.disabled = reason !== null;
        const label = button.dataset.controller === "PPO_LOCAL" ? "Local" : "Neighbor";
        button.textContent = mapName === "classroom_grid" && label === "Neighbor"
          ? "Multi-Agent RL Model" : `PPO · ${label}${data.experimental && !reason ? " (experimental)" : ""}`;
        button.title = reason || (data.experimental ? "Network-specific experimental checkpoint" : "Compatible Classroom PPO checkpoint");
      }
    } catch (error) {
      if (request !== this.learnedRequest) return;
      this.learnedReason = error.message;
      for (const button of buttons) button.title = error.message;
    } finally {
      if (request === this.learnedRequest) this.updateClassroomController();
    }
  }

  updateClassroomController() {
    const classroom = this.app.mapName === "classroom_grid";
    const active = this.currentController === MARL && !this.controllerError;
    const message = this.controllerError || this.learnedReason;
    marlNode("network-controllers").hidden = classroom;
    marlNode("marl-agent-panel").hidden = !classroom;
    marlNode("dashboard-title").textContent = classroom ? "UrbanFlow AI — Multi-Agent Traffic Control" : "City traffic dashboard";
    const button = marlNode("activate-marl");
    button.disabled = !this.app.isConnected || this.learnedReason !== null || this.modelApplying;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    const status = this.modelApplying ? "Loading trained checkpoint…"
      : message ? `Model unavailable. ${message}`
      : active ? ""
      : this.learnedReason === null ? "Checkpoint verified · Activate model" : "Checking model availability…";
    marlNode("marl-availability").textContent = status;
    marlNode("marl-availability").hidden = !status;
    if (!active) marlNode("marl-run-status").hidden = true;
    this.modelBlocksPlayback = this.controllerError || (classroom && !active);
    marlNode("btn-play-pause").disabled = !this.app.isConnected || Boolean(this.modelBlocksPlayback);
  }

  updateAgents(snapshot, ctrl) {
    if (marlNode("marl-agent-panel").hidden) return;
    const active = ctrl.active_controller === MARL && !ctrl.error;
    marlNode("marl-checkpoint").textContent = active ? ctrl.checkpoint || "Unknown" : "Inactive";
    const version = marlNode("marl-version");
    const checkpointName = ctrl.checkpoint?.match(/classroom_ppo_directional_v(\d+)\//)?.[1];
    version.hidden = !active || !ctrl.checkpoint;
    const count = ctrl.controlled_intersections_count;
    version.textContent = `${checkpointName ? `Directional v${checkpointName}` : "Original neighbor"} · ${count} Agents Loaded`;

    const decisions = active ? snapshot.signal_decisions.filter(row => row.observation) : [];
    const latest = new Map(decisions.map(row => [row.junction, row]));
    const last = ctrl.last_control_time;
    const transitions = new Map((snapshot.signal_transitions || []).map(row => [row.junction, row]));
    const direction = phase => phase === 0 ? "N/S" : phase === 1 ? "E/W" : "all red";
    const badge = marlNode("marl-run-status");
    badge.hidden = !active;
    badge.textContent = snapshot.finished ? "Demand complete" : !snapshot.paused
      ? (last == null ? "Awaiting first policy decision" : "Controlling live signals")
      : last == null ? "Ready to Start" : "Paused";
    marlNode("marl-model-details").textContent =
      `Checkpoint ${active ? "loaded" : "inactive"} · ${ctrl.policy_cycles || 0} decision cycles · ${last == null ? "Awaiting first policy decision" : `Last inference at ${last.toFixed(2)} s`}`;
    const host = marlNode("marl-agents");
    if (!host.children.length) AGENTS.forEach(([id, name], index) => {
      const card = document.createElement("article");
      const title = document.createElement("button");
      title.type = "button";
      title.className = "hud-btn";
      title.textContent = `Agent ${index + 1} · ${name}`;
      title.title = id;
      title.addEventListener("click", () => this.app.focusIntersection(id));
      card.append(title, document.createElement("p"));
      host.append(card);
    });
    const action = row => row.observation.action ? "SWITCH" : "HOLD";
    AGENTS.forEach(([id], index) => {
      const row = latest.get(id);
      const phase = snapshot.signals?.find(signal => signal.intersection_id === id)?.phase_name;
      const description = host.children[index].lastElementChild;
      const lines = [phase || "Signal state pending"];
      if (!row) lines.push("Awaiting first policy decision.");
      else {
        const obs = row.observation;
        const signal = snapshot.agent_signals?.[id];
        lines.push(`${action(row)} → ${direction(row.requested)} at ${row.time.toFixed(2)} s · ${row.outcome}`);
        if (obs.stopped) lines.push(`Queue at decision · N/S ${obs.stopped.ns} · E/W ${obs.stopped.ew} stopped`);
        if (signal?.stage === "GREEN" && signal.target != null && signal.target !== signal.active) {
          lines.push(`${direction(signal.target)} pending · eligible at ${signal.earliest_stage_end.toFixed(2)} s`);
        } else if (signal && signal.stage !== "GREEN") {
          const delay = signal.reservation ? " · junction occupied" : "";
          lines.push(`Clearance toward ${direction(signal.target)} · stage ends no earlier than ${signal.earliest_stage_end.toFixed(2)} s${delay}`);
        } else if (active) {
          const transition = transitions.get(id);
          if (transition?.stage === "GREEN") lines.push(`${direction(transition.green)} green since ${transition.time.toFixed(2)} s`);
        }
      }
      description.textContent = lines.join("\n");
    });
    const history = marlNode("marl-history");
    if (history.dataset.last === String(last) && history.dataset.active === String(active)) return;
    history.dataset.last = String(last);
    history.dataset.active = String(active);
    history.replaceChildren(...decisions.slice(-12).reverse().map(row => {
      const tr = document.createElement("tr");
      const target = row.requested === 0 ? "N/S" : "E/W";
      const result = row.safety_stage_remaining > 0
        ? `${row.outcome} · ${row.safety_stage_remaining.toFixed(1)} s left in stage` : row.outcome;
      for (const value of [`${row.time.toFixed(2)} s`, row.junction, `${action(row)} · ${target}`, result]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      return tr;
    }));
  }

  updateScenarioSelection(name) {
    document.getElementById("scenario-select").value = name;
  }

  setConnection(connected) {
    document.querySelectorAll("[data-live]").forEach(control => { control.disabled = !connected; });
    const demandUnavailable = !connected || !this.app.mapName;
    document.getElementById("scenario-select").disabled = demandUnavailable;
    document.getElementById("demand-upload").disabled = demandUnavailable;
    marlNode("route-test-start").disabled = !connected || !this.app.dashboard?.testRoute
      || Boolean(this.app.dashboard.testDriveRequest);
    this.updateClassroomController();
  }

  updateCameraSelection() {
    document.querySelectorAll("[data-cam]").forEach(button => {
      button.classList.toggle("active", button.dataset.cam === this.app.scene3d.cameraMode);
    });
  }

  inspectVehicle() {
    const manager = this.app.vehicleManager;
    const vehicle = manager.vehicles.values().next().value;
    if (vehicle) {
      manager.selectedVehicleId = vehicle.id;
      this.showVehicleDetails(vehicle.data);
      if (this.app.scene3d.cameraMode === "chase") this.app.scene3d.chaseTarget = vehicle.mesh;
    }
    return vehicle;
  }

  setFleetAvailability(hasVehicles) {
    document.getElementById("btn-inspect-fleet").disabled = !hasVehicles;
    document.querySelector('[data-cam="chase"]').disabled = !hasVehicles;
  }

  updateControllerSelection(ctrlKey) {
    document.querySelectorAll("[data-controller]").forEach(btn => {
      const match = btn.getAttribute("data-controller") === ctrlKey;
      btn.classList.toggle("active", match);
      btn.setAttribute("aria-pressed", String(match));
    });

    const displayNames = {
      "FIXED_TIME": "Fixed-Time",
      "ACTUATED": "Adaptive Actuated",
      "MAX_PRESSURE": "MaxPressure",
      "COOPERATIVE_PREDICTIVE": "CooperativePredictive",
    };

    const name = displayNames[ctrlKey] || ctrlKey;
    if (this.elCtrlActiveName) this.elCtrlActiveName.innerText = name;
    if (this.elHudCtrlName) this.elHudCtrlName.innerText = name;
  }

  updateRoutingSelection(modeKey) {
    const key = (modeKey || "OFF").toUpperCase();
    document.querySelectorAll("[data-routing]").forEach(btn => {
      const match = btn.getAttribute("data-routing") === key;
      btn.classList.toggle("active", match);
    });

    if (this.elRoutingActiveName) this.elRoutingActiveName.innerText = key;
    if (this.elHudRoutingMode) this.elHudRoutingMode.innerText = key;
  }

  // =========================================================================
  // TELEMETRY UPDATES
  // =========================================================================

  updateMetrics(metrics) {
    if (!metrics) return;
    if (this.elThroughput) this.elThroughput.innerText = Math.round(metrics.throughput_per_hour).toLocaleString();
    if (this.elAvgSpeed) this.elAvgSpeed.innerText = metrics.avg_speed_kmh.toFixed(1);
    if (this.elAvgDelay) this.elAvgDelay.innerText = metrics.avg_delay_sec.toFixed(1);
    if (this.elActiveVehicles) this.elActiveVehicles.innerText = metrics.active_vehicles;
    if (this.elArrived) this.elArrived.innerText = metrics.total_arrived || 0;

  }

  showFeedback(message, error = false) {
    const notice = document.getElementById("control-feedback");
    clearTimeout(this.noticeTimer);
    notice.textContent = message;
    notice.dataset.error = String(error);
    notice.hidden = false;
    this.noticeTimer = setTimeout(() => { notice.hidden = true; }, 4200);
  }

  setPlayback(paused, speed, time = 0, finished = false, effectiveSpeed = 0) {
    const step = document.getElementById("step-decision");
    step.disabled = !paused || finished || !this.app.isConnected || Boolean(this.modelBlocksPlayback);
    step.onclick = () => {
      step.disabled = true;
      this.app.sendCommand({action: "step_decision"});
    };
    const button = document.getElementById("btn-play-pause");
    button.dataset.paused = String(paused);
    button.textContent = finished ? "Replay" : !paused ? "Pause" : time ? "Resume" : "Start";
    document.getElementById("playback-state").textContent =
      finished ? "Demand complete" : !paused ? (effectiveSpeed && speed > 1 ? `${effectiveSpeed.toFixed(1)}× actual` : "Running") : time ? "Paused" : "Ready";
    const seconds = Math.floor(time);
    const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
    const ss = String(seconds % 60).padStart(2, "0");
    const clock = document.getElementById("sim-clock");
    clock.textContent = paused && time > 0 && !finished
      ? `${mm}:${ss}.${String(Math.round(time * 100) % 100).padStart(2, "0")}` : `${mm}:${ss}`;
    const picker = document.getElementById("playback-speed");
    // The picker also receives backend updates, including adjustments from other clients.
    if (Array.from(picker.options).some(option => Number(option.value) === speed)) picker.value = String(speed);
  }

  updateDemand(demand, revision) {
    const title = document.getElementById("study-demand");
    this.hasDemand = Boolean(demand.dataset);
    this.currentDemand = demand.dataset;
    this.trafficFactor = demand.traffic_factor || 1;
    document.getElementById("traffic-factor").value = String(this.trafficFactor);
    document.getElementById("demand-export").hidden = !this.hasDemand;
    const status = document.getElementById("demand-status");
    this.updateScenarioSelection(demand.dataset || "");
    const name = document.getElementById("scenario-select").selectedOptions[0]?.textContent
      || demand.dataset || "Procedural demand";
    const mapLabel = document.getElementById("map-select").selectedOptions[0]?.textContent || "";
    title.textContent = demand.dataset ? `${mapLabel} | ${name} | ${demand.scheduled} scheduled vehicles` : name;
    const provenance = demand.count_window ? `NYC DOT ${demand.count_window} | Simulated trips | ` : "";
    document.getElementById("count-download").hidden = !demand.count_window;
    status.textContent = demand.dataset
      ? `${provenance}${demand.scheduled} scheduled | ${demand.entered} entered | ${demand.pending} waiting to enter`
      : "Procedural demand for this network";
    if (revision !== undefined && revision !== this.demandRevision) {
      this.demandRevision = revision;
      document.getElementById("comparison-results").replaceChildren();
      if (document.getElementById("controller-study").open) this.refreshStudy();
    }
    const uploaded = demand.dataset === "uploaded";
    document.getElementById("study-preview-copy").textContent = uploaded
      ? "Replay the uploaded schedule across controllers." : "Replay the selected schedule across controllers.";
    document.getElementById("study-run").disabled = this.comparisonRunning || !this.hasDemand;
  }

  async openRlBenchmarks() {
    const dialog = document.getElementById("rl-benchmarks");
    const select = document.getElementById("rl-scenario");
    const output = document.getElementById("rl-results");
    dialog.showModal();
    output.textContent = "Loading archived SUMO results...";
    try {
      if (!this.rlStudies) {
        const response = await fetch("/api/rl-benchmarks");
        if (!response.ok) throw new Error("Benchmark archive is unavailable.");
        this.rlStudies = (await response.json()).studies;
      }
      const full = document.getElementById("rl-results-full");
      document.getElementById("rl-research").open = false;
      const render = () => {
        const study = this.rlStudies[Number(select.value)];
        this.renderSavedStudy(study, output, true);
        this.renderSavedStudy(study, full);
      };
      select.replaceChildren(...this.rlStudies.map((study, index) => new Option(study.name, index)));
      const scenario = document.getElementById("scenario-select").value;
      const selected = this.rlStudies.findIndex(study => study.id === scenario || study.id === `preset_${scenario}`);
      if (selected >= 0) select.value = String(selected);
      select.onchange = render;
      render();
    } catch (error) {
      output.textContent = error.message;
    }
  }

  renderSavedStudy(study, output, selected = false) {
    const summary = document.createElement("p");
    const results = selected ? study.results.filter(row => row.controller !== "sumo-actuated") : study.results;
    const duration = study.duration_seconds ? ` · ${study.duration_seconds} simulated seconds` : " · Duration unlisted in archive";
    const seeds = study.seeds.join(", ");
    summary.className = "study-ranking";
    summary.textContent = `${study.name} · ${study.engine} · shared seeds ${seeds}${duration}. ${study.context}.`;
    if (!results.length) {
      output.replaceChildren(summary, "No controllers in this selected view.");
      return;
    }
    const table = document.createElement("table");
    table.setAttribute("aria-label", selected ? "Selected Controller Comparison" : "Complete benchmark results");
    const header = table.createTHead().insertRow();
    for (const label of ["Controller", "Delay (s)", "Trips completed"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      header.append(cell);
    }
    const body = table.createTBody();
    for (const result of results) {
      const row = body.insertRow();
      row.classList.toggle("rl-policy-row", result.is_rl);
      row.insertCell().textContent = result.is_rl ? `${result.controller} (RL)` : result.controller;
      row.insertCell().textContent = result.delay.toFixed(2);
      row.insertCell().textContent = result.completed.toFixed(1);
    }
    const download = document.createElement("a");
    download.className = "study-download";
    download.textContent = "Download per-seed CSV";
    download.href = `/api/rl-benchmarks/${encodeURIComponent(study.source.slice(0, -4))}/csv`;
    download.download = study.source;
    output.replaceChildren(summary, table);
    if (!selected) output.append(download);
    if (study.source_url) {
      const source = document.createElement("a");
      source.className = "study-download";
      source.href = study.source_url;
      source.target = "_blank";
      source.rel = "noopener";
      source.textContent = "NYC DOT source data";
      output.append(source);
    }
  }

  openStudy(kind) {
    this.studyKind = kind;
    this.routingReportController = null;
    document.getElementById("controller-study").showModal();
    this.refreshStudy();
  }

  async refreshStudy() {
    const kind = this.studyKind;
    const routing = kind === "routing";
    const status = document.getElementById("study-status");
    const output = document.getElementById("comparison-results");
    const button = document.getElementById("study-run");
    const revision = this.demandRevision;
    document.getElementById("study-title").textContent = routing ? "Routing comparison" : "Controller comparison";
    document.getElementById("study-conditions").textContent = routing
      ? "Matched departures with the selected controller held fixed."
      : "Single schedule replay. Identical departures, empty network, routing off.";
    button.hidden = false;
    const ready = this.hasDemand;
    button.disabled = !ready || this.comparisonRunning;
    output.replaceChildren();
    status.textContent = ready ? "Ready to run a matched comparison." : "Load a supported demand schedule.";
  }

  async runComparison() {
    const kind = this.studyKind || "controller";
    if (!this.hasDemand || this.comparisonRunning) return;
    const revision = this.demandRevision;
    const matchup = kind === "routing" ? "/api/routing-comparison" : "/api/experiments/matched";
    const button = document.getElementById("study-run");
    const status = document.getElementById("study-status");
    const output = document.getElementById("comparison-results");
    this.comparisonRunning = true;
    button.disabled = true;
    output.replaceChildren();
    status.textContent = "Starting the matched replay. The live map can keep running.";
    const progress = document.getElementById("study-progress");
    progress.value = 0;
    progress.hidden = false;
    const timer = setInterval(async () => {
      try {
        const response = await fetch("/api/classroom-comparison-progress");
        const update = await response.json();
        if (this.comparisonRunning && update.kind === this.studyKind && update.controller && update.demand_revision === this.demandRevision) {
          progress.value = update.fraction;
          status.textContent = `Testing ${update.controller} | ${update.controller_number} of ${update.controller_count}`;
        }
      } catch {
        // Results or a request error will be shown when the main request finishes.
      }
    }, 1000);
    try {
      const response = await fetch(matchup);
      const report = await response.json();
      if (!response.ok) throw new Error(report.message || "Comparison could not run.");
      if (kind !== this.studyKind || revision !== this.demandRevision) return;
      if (report.demand_revision !== revision || (kind === "routing" && report.controller !== this.currentController)) {
        status.textContent = "The demand or signal strategy changed. Run the comparison again.";
        return;
      }
      status.textContent = `${report.evaluated_departures} of ${report.scheduled} departures assessed in ${report.duration_seconds} simulated seconds`;
      this.routingReportController = report.controller;
      this.renderComparison(report, output);
    } catch (error) {
      if (kind === this.studyKind && revision === this.demandRevision) status.textContent = error.message;
    } finally {
      this.comparisonRunning = false;
      clearInterval(timer);
      progress.hidden = true;
      button.disabled = !this.hasDemand || (kind === "routing" && this.currentDemand !== "uploaded");
    }
  }

  async loadHeldOut() {
    const target = document.getElementById("held-out-table");
    target.textContent = "Loading held-out results…";
    try {
      const response = await fetch("/api/experiments/held-out");
      if (!response.ok) throw new Error("Held-out results are unavailable.");
      const {summary} = await response.json();
      const names = {normal: "Normal", city_shift: "Commuter reversal", incident: "Downstream restriction"};
      const groups = Object.entries(names).map(([key, name]) => {
        const block = document.createElement("section");
        const heading = document.createElement("h4");
        heading.textContent = name;
        const rows = document.createElement("ul");
        for (const [controller, result] of Object.entries(summary[key])) {
          const item = document.createElement("li");
          const delay = result.mean_delay == null ? "—" : `${result.mean_delay.toFixed(1)} s`;
          const variation = result.delay_sd == null ? "" : ` (SD ${result.delay_sd.toFixed(1)} s)`;
          item.textContent = `${controller.replaceAll("_", " ")} · ${delay}${variation} · ${(100 * result.completion_rate).toFixed(1)}% completed`;
          rows.append(item);
        }
        block.append(heading, rows);
        return block;
      });
      target.replaceChildren(...groups);
    } catch (error) {
      target.textContent = error.message;
    }
  }

  renderComparison(report, output) {
    if (report.kind === "controller") {
      const columns = [
        ["Controller", "name"], ["Scheduled", "scheduled"], ["Completed", "completed"],
        ["Pending", "pending"], ["Active at cutoff", "active"], ["Future", "future_departures"],
        ["Removed", "removed_trips"], ["Delay mean (s)", "completed_delay_mean_sec"],
        ["Delay p95 (s)", "completed_delay_p95_sec"], ["Mean queue", "mean_queue"],
        ["Peak queue", "peak_queue"], ["Spillback (s)", "spillback_seconds"],
      ];
      const rows = [columns.map(([label]) => label),
        ...Object.values(report.results).map(result => columns.map(([, key]) => result[key]))];
      const table = document.createElement("table");
      for (const [index, values] of rows.entries()) {
        const row = (index === 0 ? table.createTHead() : table.tBodies[0] || table.createTBody()).insertRow();
        for (const value of values) {
          const cell = document.createElement(index === 0 ? "th" : "td");
          cell.textContent = value ?? "—";
          row.append(cell);
        }
      }
      const note = document.createElement("p");
      note.textContent = "Delay includes departure delay and is reported for completed trips. Read it alongside completion and cutoff counts.";
      const csv = rows.map(row => row.map(value => '"' + String(value ?? "").replaceAll('"', '""') + '"').join(",")).join("\n");
      const download = document.createElement("button");
      download.className = "hud-btn";
      download.textContent = "Download comparison CSV";
      download.addEventListener("click", () => {
        const url = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
        const link = document.createElement("a");
        link.href = url;
        link.download = "urbanflow-matched-comparison.csv";
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
      const scroller = document.createElement("div");
      scroller.style.overflowX = "auto";
      scroller.append(table);
      const chart = document.createElement("figure");
      const caption = document.createElement("figcaption");
      caption.textContent = report.restriction
        ? "Queued vehicles at the upstream junction · shaded area marks the restriction"
        : "Queued vehicles over simulated time";
      chart.append(caption);
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 480 115");
      svg.style.cssText = "width:100%;height:115px";
      const series = Object.values(report.results).map(result => ({
        name: result.name, values: report.restriction ? result.upstream_queue_trace : result.queue_trace,
      }));
      const maxQueue = Math.max(1, ...series.map(entry => Math.max(...entry.values)));
      const x = time => 478 * time / report.duration_seconds;
      if (report.restriction) {
        const band = document.createElementNS(svg.namespaceURI, "rect");
        band.setAttribute("x", x(report.restriction.start));
        band.setAttribute("width", x(report.restriction.end - report.restriction.start));
        band.setAttribute("height", "110");
        band.setAttribute("fill", "#f1b75f35");
        svg.append(band);
      }
      const legend = document.createElement("p");
      for (const [index, entry] of series.entries()) {
        const color = ["#67e8f9", "#f5bd65", "#ad99fb", "#80e1a0", "#ff9292", "#a6aee1"][index];
        const line = document.createElementNS(svg.namespaceURI, "polyline");
        line.setAttribute("points", entry.values.map((value, i) => `${x(i + 1)},${110 - 105 * value / maxQueue}`).join(" "));
        line.setAttribute("fill", "none");
        line.setAttribute("stroke", color);
        line.setAttribute("stroke-width", "1.5");
        svg.append(line);
        const label = document.createElement("span");
        label.style.color = color;
        label.textContent = `${entry.name} · `;
        legend.append(label);
      }
      chart.append(svg, legend);
      output.replaceChildren(note, chart, scroller, download);
      return;
    }
    const summary = document.createElement("section");
    summary.className = "study-recommendation";
    const heading = document.createElement("h3");
    const detail = document.createElement("p");
    const winner = report.results[report.recommended];
    const routing = report.kind === "routing";
    const baseline = routing ? report.results.OFF : report.results.FIXED_TIME;
    if (winner) {
      heading.textContent = `Recommended for this demand: ${winner.name}`;
      const second = report.results[report.ranked[1]];
      const evidence = winner.completed > second.completed
        ? `${winner.completed} trips completed, ${winner.completed - second.completed} more than the next option.`
        : winner.mean_queue < second.mean_queue
          ? `Tied for trips completed and had a shorter average queue: ${winner.mean_queue} vs ${second.mean_queue} vehicles.`
          : `Tied on completions and average queue, with lower completed-trip waiting: ${winner.wait_sec} vs ${second.wait_sec} seconds.`;
      const delta = winner.completed - baseline.completed;
      detail.textContent = `${evidence} ${delta ? `${delta} more completions than ${baseline.name}.` : winner.description}`;
      const apply = document.createElement("button");
      apply.className = "hud-btn";
      apply.textContent = `Use ${winner.name}`;
      apply.addEventListener("click", async () => {
        apply.disabled = true;
        const command = routing
          ? { action: "set_routing_mode", mode: report.recommended, comparison_controller: report.controller }
          : { action: "set_controller", controller: report.recommended };
        command.demand_revision = report.demand_revision;
        const ok = await this.app.sendCommand(command, `${winner.name} applied to the live simulation.`);
        apply.textContent = ok ? `${winner.name} applied` : `Use ${winner.name}`;
        apply.disabled = ok;
        if (ok) document.getElementById("study-status").textContent =
          `${winner.name} applied. Check Controls for the next live update.`;
      });
      summary.append(heading, detail);
      if (routing || this.app.mapName !== "classroom_grid" || report.recommended === MARL) summary.append(apply);
    } else {
      heading.textContent = "No clear recommendation";
      detail.textContent = routing && Object.values(report.results).every(item => item.reroutes === 0)
        ? "No reroutes were accepted in these replays, so the modes used the same routes."
        : "The top options tied on the ranking measures, or no trips completed in the measured window.";
      summary.append(heading, detail);
    }

    const ranking = document.createElement("p");
    ranking.className = "study-ranking";
    ranking.textContent = "Ranking favors more completed trips, then fewer queued vehicles, then less stopped time among completed trips.";
    const table = document.createElement("table");
    table.setAttribute("aria-label", routing ? "Measured routing performance" : "Measured controller performance");
    const header = table.createTHead().insertRow();
    const columns = [routing ? "Routing mode" : "Controller", "Trips", "Avg. queue", "Trip wait", "Waiting to enter"];
    if (routing) columns.push("Reroutes");
    for (const label of columns) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      header.append(cell);
    }
    const maxTrips = Math.max(1, ...Object.values(report.results).map(item => item.completed));
    const body = table.createTBody();
    for (const name of report.ranked) {
      const item = report.results[name];
      const row = body.insertRow();
      const label = document.createElement("th");
      label.scope = "row";
      label.textContent = item.name;
      const description = document.createElement("small");
      description.textContent = item.predictive_status && item.predictive_status !== "ACTIVE"
        ? `${item.description} Forecast status: ${item.predictive_status.toLowerCase().replaceAll("_", " ")}.`
        : item.description;
      label.append(description);
      row.append(label);
      const trips = row.insertCell();
      trips.textContent = item.completed;
      const bar = document.createElement("meter");
      bar.min = 0;
      bar.max = maxTrips;
      bar.value = item.completed;
      bar.setAttribute("aria-label", `${item.name} completed trips`);
      trips.append(bar);
      row.insertCell().textContent = item.mean_queue.toFixed(1);
      row.insertCell().textContent = item.wait_sec === null ? "-" : `${item.wait_sec.toFixed(1)} s`;
      row.insertCell().textContent = item.never_entered;
      if (routing) row.insertCell().textContent = item.reroutes;
    }
    output.replaceChildren(summary, ranking, table);
  }

  updateSignals(signalStates) {
    if (!signalStates || signalStates.length === 0) return;
    const s = signalStates.find(signal => signal.intersection_id === this.app.focusedIntersectionId) || signalStates[0];

    if (this.elSignalName) this.elSignalName.innerText = `${s.intersection_id} | ${s.phase_name}`;
    if (this.elSignalTimer) {
      this.elSignalTimer.innerText = `${s.time_in_phase.toFixed(1)} s in phase`;
    }

    const isGreen = s.state_string.includes("G");
    const isYellow = s.state_string.includes("y");
    const isRed = s.state_string.includes("r") && !isGreen && !isYellow;

    if (this.dotRed) this.dotRed.classList.toggle("active", isRed);
    if (this.dotYellow) this.dotYellow.classList.toggle("active", isYellow);
    if (this.dotGreen) this.dotGreen.classList.toggle("active", isGreen);

    const usePicker = signalStates.length > 8;
    document.getElementById("junction-picker").hidden = !usePicker;
    this.elBottleneckList.hidden = usePicker;
    if (usePicker) {
      const ids = signalStates.map(signal => signal.intersection_id);
      const signature = ids.join("|");
      const select = document.getElementById("junction-select");
      if (this.junctionChoices !== signature) {
        this.junctionChoices = signature;
        select.replaceChildren(new Option(`Choose one of ${ids.length} junctions`, ""),
          ...ids.map(id => new Option(this.app.intersections?.find(item => item.id === id)?.name || id, id)));
      }
      select.value = this.app.focusedIntersectionId || "";
      document.getElementById("junction-phase").textContent = `${s.intersection_id} | ${s.phase_name}`;
      return;
    }

    if (this.elBottleneckList) {
      const summary = signalStates.map(sig => [sig.intersection_id, sig.phase_name]);
      const signature = JSON.stringify(summary);
      if (this.signalSummary !== signature) {
        this.signalSummary = signature;
        this.elBottleneckList.replaceChildren(...summary.map(([id, phase]) => {
          const button = document.createElement("button");
          button.className = "intersection-row hud-btn";
          button.textContent = `${id} | ${phase}`;
          button.addEventListener("click", () => this.app.focusIntersection(id));
          return button;
        }));
      }
    }
  }

  updateSignalController(ctrlData) {
    if (!ctrlData) return;
    const activeKey = ctrlData.active_controller || "FIXED_TIME";
    this.currentController = activeKey;
    this.controllerError = ctrlData.error;
    this.currentControllerName = ctrlData.controller_name || activeKey;
    if (this.studyKind === "routing" && this.routingReportController && this.routingReportController !== activeKey) {
      this.routingReportController = null;
      document.getElementById("comparison-results").replaceChildren();
      document.getElementById("study-status").textContent = "The signal controller changed. Run a fresh routing comparison.";
      document.getElementById("study-conditions").textContent =
        `Same departures, empty network. Signal controller held fixed: ${this.currentControllerName}.`;
    }
    this.updateControllerSelection(activeKey);

    if (this.elCtrlActiveName) this.elCtrlActiveName.innerText = ctrlData.controller_name || activeKey;
    if (this.elHudCtrlName) this.elHudCtrlName.innerText = ctrlData.controller_name || activeKey;

    document.getElementById("controller-explanation").textContent = ctrlData.description || "";
    if (this.elCtrlModelTag) {
      this.elCtrlModelTag.innerText = ctrlData.last_control_time == null
        ? "Awaiting first decision" : `Decision at ${ctrlData.last_control_time.toFixed(1)} s`;
    }

    if (this.elHudCtrlSignals && ctrlData.controlled_intersections_count !== undefined) {
      this.elHudCtrlSignals.innerText = `${ctrlData.controlled_intersections_count} Signals`;
    }
    this.updateClassroomController();
  }

  updateRoutingController(routingData) {
    if (!routingData) return;
    const activeMode = (routingData.active_mode || "OFF").toUpperCase();
    this.updateRoutingSelection(activeMode);
    document.getElementById("routing-explanation").textContent = routingData.mode_description || "";

    if (this.elRoutingActiveName) this.elRoutingActiveName.innerText = routingData.mode_name || activeMode;
    if (this.elHudRoutingMode) this.elHudRoutingMode.innerText = routingData.mode_name || activeMode;
    if (this.elRoutingModelTag) this.elRoutingModelTag.textContent = activeMode === "OFF" ? "Off"
      : routingData.last_evaluation_time == null ? "Awaiting first check"
        : `Checked at ${routingData.last_evaluation_time.toFixed(1)} s`;

    if (this.elValRerouteCount) this.elValRerouteCount.innerText = routingData.accepted_count || 0;
    if (this.elValRerouteRejected) this.elValRerouteRejected.innerText = routingData.rejected_count || 0;
    if (this.elValReservationsCount) this.elValReservationsCount.innerText = routingData.active_reservations_count || 0;
  }

  updatePredictiveIntelligence(intelData) {
    if (!intelData) return;
  }

  showVehicleDetails(vData) {
    if (!this.inspector || !vData) return;
    this.inspector.style.display = "flex";

    if (this.inspId) this.inspId.innerText = vData.id;
    if (this.inspType) this.inspType.innerText = (vData.type || "sedan").toUpperCase();
    if (this.inspSpeed) this.inspSpeed.innerText = `${(vData.speed * 3.6).toFixed(1)} km/h`;
    if (this.inspRoad) this.inspRoad.innerText = vData.road_id || "-";
    if (this.inspRouteProgress) this.inspRouteProgress.innerText = `${(vData.route_index || 0) + 1} / ${(vData.route || []).length}`;
    if (this.inspRerouteCount) this.inspRerouteCount.innerText = vData.reroute_count || 0;
    if (this.inspRouteGuidance) {
      const isRerouted = (vData.reroute_count || 0) > 0;
      this.inspRouteGuidance.innerText = isRerouted ? "Diverted" : "Original";
      this.inspRouteGuidance.style.color = isRerouted ? "#10b981" : "#fff";
    }
    if (this.inspWait) this.inspWait.innerText = `${(vData.wait_time || 0).toFixed(1)}s`;

    if (this.app.roadBuilder) {
      this.app.roadBuilder.highlightRoute(vData.route, vData.original_route, vData.road_id);
    }
  }
}
