import { VehicleManager } from "./vehicles.js?v=route-traffic-1";

const COLORS = ["#45dc8a", "#f0ca55", "#f59851", "#ff626f"];
const number = new Intl.NumberFormat();
const get = id => document.getElementById(id);

function clock(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

export class Dashboard {
  constructor(app) {
    this.app = app;
    this.history = [];
    this.markers = [];
    this.searchItems = [];
    this.roads = new Map();
    this.plannedRoute = null;
    this.labels = get("map-labels");
    this.trend = get("traffic-trend");
    this.markerPoint = new THREE.Vector3();
    this.bindControls();
    this.resizeObserver = new ResizeObserver(() => this.drawTrend());
    this.resizeObserver.observe(this.trend);
    app.scene3d.onRender(() => this.projectMarkers());
    app.scene3d.onRender((dt, elapsed) => this.replayTraffic?.animate(dt, elapsed));
  }

  bindControls() {
    document.querySelectorAll("[data-open]").forEach(button => {
      button.addEventListener("click", () => {
        const dialog = get(button.dataset.open);
        if (!dialog.open) button.dataset.open === "signals-dialog" ? dialog.show() : dialog.showModal();
      });
    });
    get("map-zoom-in").addEventListener("click", () => this.app.scene3d.zoom(.76));
    get("map-zoom-out").addEventListener("click", () => this.app.scene3d.zoom(1.31));
    get("map-fit").addEventListener("click", () => {
      this.setTestCarCamera(false);
      this.app.scene3d.setCameraView("tactical");
      this.app.hud.updateCameraSelection();
    });
    const mapPanel = get("map-panel");
    const fullscreen = get("map-fullscreen");
    const updateFullscreen = () => {
      const expanded = document.fullscreenElement === mapPanel || mapPanel.classList.contains("map-expanded");
      fullscreen.setAttribute("aria-pressed", String(expanded));
      fullscreen.setAttribute("aria-label", expanded ? "Exit fullscreen map" : "Fullscreen map");
    };
    document.addEventListener("fullscreenchange", updateFullscreen);
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && mapPanel.classList.contains("map-expanded")) {
        mapPanel.classList.remove("map-expanded");
        updateFullscreen();
      }
    });
    fullscreen.addEventListener("click", async () => {
      try {
        if (document.fullscreenElement === mapPanel) await document.exitFullscreen();
        else if (mapPanel.classList.contains("map-expanded")) mapPanel.classList.remove("map-expanded");
        else if (document.fullscreenEnabled) await mapPanel.requestFullscreen();
        else mapPanel.classList.add("map-expanded");
      } catch {
        mapPanel.classList.toggle("map-expanded");
      }
      updateFullscreen();
    });
    get("lighting-toggle").addEventListener("click", event => {
      const night = !this.app.scene3d.isNightMode;
      this.app.scene3d.setNightMode(night);
      this.app.roadBuilder.setNightMode(night);
      event.currentTarget.textContent = night ? "Night" : "Day";
      event.currentTarget.setAttribute("aria-pressed", String(night));
    });
    get("map-search").addEventListener("submit", event => {
      event.preventDefault();
      const query = get("map-search-input").value.trim().toLowerCase();
      const match = query && (this.searchItems.find(item => item.label.toLowerCase() === query)
        || this.searchItems.find(item => item.label.toLowerCase().includes(query)));
      if (!match) return this.app.hud.showFeedback("No road or junction matches this search.", true);
      this.setTestCarCamera(false);
      this.app.scene3d.focusIntersection(match.position.x, match.position.y);
      this.app.hud.updateCameraSelection();
      if (match.junction) this.app.focusedIntersectionId = match.junction;
    });
    get("trend-window").addEventListener("change", () => this.drawTrend());
    get("route-swap").addEventListener("click", () => {
      const from = get("route-from"), to = get("route-to");
      [from.value, to.value] = [to.value, from.value];
    });
    get("route-form").addEventListener("submit", event => {
      event.preventDefault();
      this.findRoute();
    });
    get("route-highlight-toggle").addEventListener("click", () => {
      this.setPlannedRouteHighlight(!this.showPlannedRoute);
    });
    window.addEventListener("hashchange", () => {
      if (location.hash !== "#route-panel") this.setPlannedRouteHighlight(false);
    });
    get("route-test-start").addEventListener("click", () => this.runTestDrive());
    const disclosure = get("route-test-panel");
    get("route-test-close").addEventListener("click", () => {
      disclosure.open = false;
      disclosure.querySelector("summary").focus();
    });
    disclosure.addEventListener("toggle", () => {
      disclosure.querySelector("summary").textContent = disclosure.open ? "Close Route Test Drive" : "Open Route Test Drive";
      if (!disclosure.open) {
        this.setPlannedRouteHighlight(false);
        this.setTestCarCamera(false);
        if (this.testCar) this.testCar.visible = false;
      } else if (!this.testRoute && this.app.isConnected) {
        this.findRoute(null, false);
      }
    });
    get("route-chase-exit").addEventListener("click", () => {
      this.setTestCarCamera(false);
      get("route-test-panel").scrollIntoView({block: "nearest"});
    });
    get("route-chase-return").addEventListener("click", () => this.setTestCarCamera(true));
    for (const id of ["route-from", "route-to", "route-swap"]) {
      get(id).addEventListener(id === "route-swap" ? "click" : "change", () => {
        this.clearTestDrive();
        if (!this.plannedRoute) return;
        this.plannedRoute = null;
        this.setPlannedRouteHighlight(false);
        get("route-result").textContent = "Endpoints changed. Find the new route.";
      });
    }
    get("routing-demo").addEventListener("click", async event => {
      const button = event.currentTarget;
      button.disabled = true;
      try {
        const result = await this.app.sendCommand({action: "start_routing_demo"});
        const demo = result?.demo;
        if (!demo || demo.map_name !== this.app.mapName) return;
        get("route-from").value = demo.from;
        get("route-to").value = demo.to;
        const road = this.roads.get(demo.road);
        await this.findRoute(road?.name || demo.road);
      } finally {
        button.disabled = !this.app.isConnected;
      }
    });
    get("incident-form").addEventListener("submit", event => {
      event.preventDefault();
      const road = get("incident-road").value;
      if (road) this.app.sendCommand({action: "inject_incident", road_id: road, severity: .65, duration: 90},
        "Incident added for 90 simulated seconds.");
    });
  }

  setTopology(topology) {
    this.roads = new Map(topology.roads.map(road => [road.id, road]));
    this.totalRoads = this.roads.size;
    const junctions = topology.intersections;
    this.networkName = topology.metadata?.name || get("map-select").selectedOptions[0]?.textContent || "Road network";
    const signals = junctions.filter(junction => junction.has_signals);
    this.routeJunctions = new Set(signals.map(junction => junction.id));
    get("network-size").textContent = `${signals.length} signals / ${this.totalRoads} directed roads`;
    get("dashboard-network").textContent = `${this.networkName} / Live simulation`;
    get("road-count").textContent = this.totalRoads;
    get("map-search-input").value = "";
    this.history = [];
    this.lastTime = null;
    this.plannedRoute = null;
    this.setPlannedRouteHighlight(false);
    this.clearTestDrive();
    get("route-result").textContent = "Estimated travel time uses current link speeds.";
    get("route-result").dataset.error = "false";
    const nodes = new Map(junctions.map(junction => [junction.id, junction.name || junction.id]));
    const corridors = new Map();
    const center = this.app.scene3d.networkCenter;
    for (const road of this.roads.values()) {
      for (const id of [road.from_node, road.to_node]) {
        if (!nodes.has(id)) nodes.set(id, id.replaceAll("_", " "));
      }
      if (!road.points.length) continue;
      const label = road.name || road.id;
      const middle = road.points[Math.floor(road.points.length / 2)];
      const distance = (point) => Math.hypot(point.x - center.x, point.y - center.z);
      if (!corridors.has(label) || distance(middle) < distance(corridors.get(label))) corridors.set(label, middle);
    }
    for (const id of ["route-from", "route-to"]) {
      get(id).replaceChildren(...Array.from(nodes, ([value, label]) => new Option(label, value)));
    }
    if (topology.route_defaults) {
      get("route-from").value = topology.route_defaults.from;
      get("route-to").value = topology.route_defaults.to;
    } else {
      get("route-to").selectedIndex = Math.min(1, nodes.size - 1);
    }
    get("incident-road").replaceChildren(...topology.roads.map(road => new Option(
      road.name ? `${road.name} (${road.from_node} / ${road.to_node})` : road.id, road.id)));
    this.searchItems = [...junctions.map(junction => ({label: junction.name || junction.id,
      position: junction.position, junction: junction.id})),
    ...Array.from(corridors, ([label, position]) => ({label, position}))];
    get("map-search-options").replaceChildren(...this.searchItems.map(item => {
      const option = document.createElement("option");
      option.value = item.label;
      return option;
    }));
    let marked = signals;
    if (signals.length > 8) {
      const xs = signals.map(({position}) => position.x), ys = signals.map(({position}) => position.y);
      const minX = Math.min(...xs), minY = Math.min(...ys);
      const spanX = Math.max(...xs) - minX, spanY = Math.max(...ys) - minY;
      marked = [...new Set([[.2,.2],[.77,.22],[.46,.50],[.23,.76],[.8,.71],[.55,.9]].map(([x, y]) => {
        const score = ({position}) => Math.hypot(position.x - minX - x * spanX, position.y - minY - y * spanY);
        return signals.reduce((best, item) => score(item) < score(best) ? item : best);
      }))];
    }
    this.labels.replaceChildren();
    this.markers = marked.map(junction => {
      const node = document.createElement("span");
      node.className = "map-label signal";
      node.textContent = junction.name || junction.id;
      this.labels.append(node);
      return {id: junction.id, node, position: junction.position};
    });
    this.updateDistribution({});
    this.updateIncidents([], 0);
    this.drawTrend();
  }

  projectMarkers() {
    const {camera, controls, cameraMode} = this.app.scene3d;
    const width = this.labels.clientWidth, height = this.labels.clientHeight;
    const occupied = [];
    for (const {node, position, landmark} of this.markers) {
      const point = this.markerPoint.set(position.x, 5, position.y).project(camera);
      const x = (point.x / 2 + .5) * width, y = (.5 - point.y / 2) * height;
      node.hidden = (!landmark && width < 460) || cameraMode === "chase" || point.z >= 1 || y < 53 || y > height - 30 || x < 67 || x > width - 67
        || occupied.some(([cx, cy]) => Math.abs(cx - x) < 140 && Math.abs(cy - y) < 25);
      if (!node.hidden) {
        occupied.push([x, y]);
        node.style.left = `${x}px`;
        node.style.top = `${y}px`;
      }
    }
    const scale = get("map-scale");
    scale.hidden = cameraMode !== "tactical" || width < 300;
    if (!scale.hidden) {
      const meters = camera.position.distanceTo(controls.target) * 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
      const length = [5, 10, 20, 50, 100, 200, 500, 1000, 2000].findLast(value => value * height / meters <= 110);
      if (length) {
        scale.style.width = `${length * height / meters}px`;
        scale.textContent = length >= 1000 ? `${length / 1000} km` : `${length} m`;
      }
    }
  }

  update(snapshot, intelligence) {
    const time = snapshot.timestamp;
    const junction = this.app.focusedIntersectionId || snapshot.signals?.[0]?.intersection_id;
    const recent = (snapshot.signal_decisions || []).filter(item => item.junction === junction).slice(-4);
    const decision = recent.at(-1);
    const phase = snapshot.signals?.find(item => item.intersection_id === junction);
    const directions = {0: "north–south", 1: "east–west", 2: "east–west", "-1": "all-red hold", 4: "all-red hold"};
    get("junction-decision").textContent = decision
      ? `At ${decision.time} s: requested ${directions[decision.requested] || "an invalid phase"}. ${decision.outcome}. Current simulator stage: ${phase?.phase_name || decision.stage}.`
      : "No requests recorded for this junction yet.";
    const observations = decision?.observation;
    const transitions = (snapshot.signal_transitions || []).filter(item => item.junction === junction).slice(-5);
    get("junction-observations").textContent = observations
      ? [`P(hold): ${observations.probabilities[0].toFixed(3)} · P(switch): ${observations.probabilities[1].toFixed(3)}`,
        ...recent.map(item => `${item.time} s · request ${directions[item.requested] || "invalid"} · ${item.outcome}`),
        ...transitions.map(item => `${item.time} s · ${item.stage}${item.stage === "GREEN" ? " " + directions[item.green] : ""}${item.reason ? " · " + item.reason : ""}`),
        ...Object.entries(observations.features).map(([key, value]) => `${key}: ${value.toFixed(3)}`)].join("\n")
      : "";
    get("live-congestion").textContent = `${Math.round((snapshot.metrics?.congestion_index || 0) * 100)}%`;
    const demand = snapshot.demand;
    if (demand) {
      const selection = get("scenario-select").selectedOptions[0]?.textContent || "Current demand";
      get("dashboard-scenario").textContent = `${selection} / ${number.format(demand.scheduled)} scheduled`;
      get("departure-progress").value = demand.scheduled ? demand.due / demand.scheduled : 0;
      get("departure-summary").textContent = `${number.format(demand.due)} / ${number.format(demand.scheduled)} due`;
    }
    if (time < this.lastTime || snapshot.demand_revision !== this.demandRevision) {
      this.history = [];
      this.lastTime = null;
      this.updateDistribution({});
      this.drawTrend();
    }
    this.demandRevision = snapshot.demand_revision;
    if (this.lastTime == null || time - this.lastTime >= 1) {
      this.history.push([time, snapshot.vehicles?.length || 0]);
      this.history = this.history.filter(([at]) => at >= time - 1800);
      this.lastTime = time;
      this.drawTrend();
    }
    this.updateIncidents(snapshot.incidents || [], time);
    const signals = new Map(snapshot.signals?.map(signal => [signal.intersection_id, signal.stage]) || []);
    for (const {id, node} of this.markers) node.dataset.stage = signals.get(id) || "";
    if (intelligence?.road_heatmap) this.updateDistribution(intelligence.road_heatmap);
  }

  updateDistribution(heatmap) {
    const bins = [0, 0, 0, 0], colors = {}, corridors = new Map();
    for (const road of this.roads.values()) {
      const state = heatmap[road.id] || {};
      const ratio = state.vc_ratio || 0;
      const band = ratio < .30 ? 0 : ratio < .60 ? 1 : ratio < .85 ? 2 : 3;
      bins[band]++;
      colors[road.id] = {color: COLORS[band]};
      if (state.vehicle_count > 0) {
        const name = road.name || road.id;
        const item = corridors.get(name) || {count: 0, sum: 0, limit: 0};
        item.count += state.vehicle_count;
        item.sum += state.speed * state.vehicle_count;
        item.limit = Math.max(item.limit, road.speed_limit * 3.6);
        corridors.set(name, item);
      }
    }
    this.app.roadBuilder.updateRoadHeatmap(colors);
    let running = 0;
    const stops = bins.map((count, i) => {
      const start = running;
      const percentage = this.totalRoads ? count / this.totalRoads * 100 : 0;
      get(`density-${i}`).textContent = `${percentage.toFixed(0)}%`;
      running += percentage;
      return `${COLORS[i]} ${start}% ${Math.min(100, running)}%`;
    });
    get("traffic-donut").style.background = `conic-gradient(${stops.join(", ")})`;
    const list = get("corridor-speeds");
    const rows = [...corridors].sort((a, b) => b[1].count - a[1].count).slice(0, 4);
    list.replaceChildren(...rows.map(([name, item]) => {
      const li = document.createElement("li"), title = document.createElement("span");
      const meter = document.createElement("meter"), reading = document.createElement("strong");
      const speed = item.sum / item.count;
      title.textContent = title.title = name;
      meter.max = Math.max(item.limit, speed, 1);
      meter.value = speed;
      meter.setAttribute("aria-label", `${name} average speed`);
      reading.textContent = `${speed.toFixed(1)} km/h`;
      li.append(title, meter, reading);
      return li;
    }));
    if (!rows.length) {
      const li = document.createElement("li");
      li.textContent = "No observed corridor speeds yet";
      list.append(li);
    }
  }

  updateIncidents(incidents, time) {
    get("live-incidents").textContent = incidents.length;
    get("incident-brief").textContent = incidents.length ? "Simulated road restrictions" : "No active incidents";
    if (!get("incident-dialog").open && !incidents.length && this.incidentCount === 0) return;
    this.incidentCount = incidents.length;
    const rows = incidents.length ? incidents.map(incident => {
      const road = this.roads.get(incident.road_id);
      return `${road?.name || incident.road_id} / ${Math.ceil(incident.end_time - time)} s remaining`;
    }) : ["No active incidents"];
    get("incident-list").replaceChildren(...rows.map(label => {
      const li = document.createElement("li");
      li.textContent = label;
      return li;
    }));
  }

  drawTrend() {
    const canvas = this.trend;
    const {clientWidth: width, clientHeight: height} = canvas;
    if (width < 60 || height < 60) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const right = width - 10, left = 32, bottom = height - 22, top = 11;
    const end = this.history.at(-1)?.[0] || 0, start = Math.max(0, end - Number(get("trend-window").value));
    const points = this.history.filter(([at]) => at >= start);
    const max = Math.max(4, Math.ceil(Math.max(0, ...points.map(([, count]) => count)) / 4) * 4);
    ctx.font = "10px Inter, sans-serif";
    ctx.fillStyle = "#9fb5ca";
    ctx.strokeStyle = "#26384c";
    for (let i = 0; i <= 4; i++) {
      const y = bottom - (bottom - top) * i / 4;
      ctx.fillText(String(max * i / 4), 1, y + 3);
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    }
    const span = Math.max(1, end - start);
    ctx.fillText(clock(start), left, height - 4);
    if (end > start) {
      const label = clock(end);
      ctx.fillText(label, right - ctx.measureText(label).width, height - 4);
    }
    if (!points.length) return;
    const line = points.map(([at, count]) => [left + (at - start) / span * (right - left), bottom - count / max * (bottom - top)]);
    const area = ctx.createLinearGradient(0, top, 0, bottom);
    area.addColorStop(0, "#4ce9e860"); area.addColorStop(1, "#4ce9e808");
    ctx.beginPath(); ctx.moveTo(line[0][0], bottom);
    for (const [x, y] of line) ctx.lineTo(x, y);
    ctx.lineTo(line.at(-1)[0], bottom); ctx.closePath(); ctx.fillStyle = area; ctx.fill();
    ctx.beginPath(); ctx.moveTo(...line[0]);
    for (const [x, y] of line.slice(1)) ctx.lineTo(x, y);
    ctx.strokeStyle = "#63eef1"; ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath(); ctx.arc(...line.at(-1), 3, 0, Math.PI * 2); ctx.fillStyle = "#a7ffff"; ctx.fill();
  }

  setPlannedRouteHighlight(visible) {
    this.showPlannedRoute = visible;
    const button = get("route-highlight-toggle");
    button.hidden = !this.plannedRoute;
    button.textContent = visible ? "Hide route on map" : "Show route on map";
    button.setAttribute("aria-pressed", String(visible && Boolean(this.plannedRoute)));
    if (!this.app.vehicleManager.selectedVehicleId) {
      this.app.roadBuilder.highlightRoute(visible ? this.plannedRoute : null);
    }
  }

  async findRoute(demoRoad = null, highlight = true) {
    const output = get("route-result"), map = this.app.mapName;
    this.clearTestDrive();
    this.plannedRoute = null;
    this.setPlannedRouteHighlight(highlight);
    const version = this.routeTestVersion;
    const from = get("route-from").value, to = get("route-to").value;
    output.dataset.error = "false";
    output.textContent = "Finding route";
    try {
      const response = await fetch("/api/route-plan", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({map_name: map, from, to})});
      const data = await response.json();
      if (this.app.mapName !== map || version !== this.routeTestVersion) return;
      if (!response.ok) throw new Error(data.message || "Route unavailable.");
      this.plannedRoute = data.road_ids;
      this.testRoute = {map_name: map, from, to, road_ids: data.road_ids};
      get("route-test-start").disabled = !this.app.isConnected;
      const label = id => get(id).selectedOptions[0]?.textContent || get(id).value;
      get("route-test-endpoints").textContent = `Start · ${label("route-from")}  →  Destination · ${label("route-to")}`;
      this.makeRoutePreview()(null);
      get("close-inspector").click();
      this.setPlannedRouteHighlight(this.showPlannedRoute);
      const time = data.estimated_seconds < 60 ? `${data.estimated_seconds} s` : `${(data.estimated_seconds / 60).toFixed(1)} min`;
      const direct = data.shortest;
      const directTime = direct.estimated_seconds < 60 ? `${direct.estimated_seconds} s` : `${(direct.estimated_seconds / 60).toFixed(1)} min`;
      const saving = Math.max(0, direct.estimated_seconds - data.estimated_seconds);
      const detail = saving >= 5 || (demoRoad && saving > 0) ? ` Saves ${Math.round(saving)} s versus the direct route.`
        : ` Includes ${data.signal_wait_seconds} s signal delay and ${data.queue_delay_seconds} s queue delay.`;
      output.textContent = `${(data.distance_m / 1000).toFixed(2)} km / ${time} estimated / ${data.average_speed_kmh.toFixed(1)} km/h. Direct: ${(direct.distance_m / 1000).toFixed(2)} km / ${directTime}.${detail}`;
      if (demoRoad) {
        output.textContent += ` Demo restriction on ${demoRoad}.`;
        if (saving === 0) output.textContent += " No faster detour for this trip.";
      }
    } catch (error) {
      if (this.app.mapName !== map || version !== this.routeTestVersion) return;
      output.dataset.error = "true";
      output.textContent = error.message;
    }
  }

  clearTestDrive() {
    this.testDriveRequest?.abort();
    this.testDriveRequest = null;
    this.removeTestCar();
    this.routeTestVersion = (this.routeTestVersion || 0) + 1;
    this.testRoute = null;
    get("route-test-map").hidden = true;
    get("route-test-status").textContent = "";
    get("route-test-results").replaceChildren();
    get("route-test-start").disabled = true;
    get("route-test-endpoints").textContent = "Choose a route in the Route planner.";
  }

  async runTestDrive() {
    const version = ++this.routeTestVersion;
    const status = get("route-test-status"), button = get("route-test-start");
    if (!this.testRoute || !this.app.isConnected || this.testDriveRequest) return;
    const request = new AbortController();
    this.testDriveRequest = request;
    this.removeTestCar();
    button.disabled = true;
    this.makeRoutePreview()(null);
    get("route-test-results").replaceChildren();
    status.textContent = "Starting the test drive…";
    const replay = {trace: [], traffic: [], finished: false, result: null, timeout: null};
    try {
      const response = await fetch("/api/experiments/route?stream=1", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...this.testRoute, session: this.testDriveSession ||= `${Date.now()}-${Math.random()}`}),
        signal: request.signal});
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || "The test drive could not run.");
      }
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "", comparison, playback;
      try {
        while (true) {
          const {value, done} = await reader.read();
          if (done || version !== this.routeTestVersion) break;
          buffer += value;
          const lines = buffer.split("\n");
          buffer = lines.pop();
          for (const line of lines) {
            const message = JSON.parse(line);
            if (message.type === "error") throw new Error(message.message);
            if (message.type === "comparison") comparison = message.report;
            if (message.type === "samples") {
              replay.trace.push(...message.report.trace);
              replay.traffic.push(...message.report.traffic);
            }
            if (message.type === "replay") {
              replay.finished = true;
              replay.result = message.report.results[0];
              replay.timeout = message.report.timeout_seconds;
            }
            if (!playback && (replay.trace.length > 1 || replay.finished)) {
              playback = this.playTestDrive(replay, version).then(() => {
                if (version === this.routeTestVersion && !comparison) {
                  status.textContent = "Replay complete. Comparing the remaining controllers…";
                }
              });
            }
          }
        }
      } finally {
        if (!request.signal.aborted) await reader.cancel();
      }
      if (version !== this.routeTestVersion) return;
      if (!comparison) throw new Error("The comparison ended before results arrived. Try again.");
      await playback;
      if (version === this.routeTestVersion) this.showTestResults(comparison);
    } catch (error) {
      if (version === this.routeTestVersion) {
        this.removeTestCar();
        status.textContent = error.message;
        if (get("route-test-panel").open) get("route-test-panel").scrollIntoView({block: "nearest"});
      }
    } finally {
      if (this.testDriveRequest === request) this.testDriveRequest = null;
      if (version === this.routeTestVersion) button.disabled = !this.app.isConnected;
    }
  }

  removeTestCar() {
    this.setTestCarCamera(false);
    this.replayTraffic?.dispose();
    this.replayTraffic = null;
    this.replaySignals = null;
    if (this.testCar) {
      this.app.scene3d.scene.remove(this.testCar);
      const materials = new Set();
      for (const mesh of this.testCar.children) {
        mesh.geometry.dispose();
        materials.add(mesh.material);
      }
      for (const material of materials) material.dispose();
      this.testCar = null;
    }
    get("route-chase-return").hidden = true;
  }

  setTestCarCamera(follow) {
    if (follow && !this.testCar) return;
    const view = this.app.scene3d;
    if (follow && !this.testCarCamera) {
      this.testCar.visible = true;
      this.testCarCamera = {mode: view.cameraMode, target: view.chaseTarget,
        position: view.camera.position.clone(), lookAt: view.controls.target.clone()};
      this.app.vehicleManager.vehicleGroup.visible = false;
      if (this.replayTraffic) this.replayTraffic.vehicleGroup.visible = true;
      if (this.replaySignals) this.app.roadBuilder.updateSignals(this.replaySignals);
      view.setCameraView("chase", this.testCar);
      const offset = new THREE.Vector3(0, 5, -12).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.testCar.rotation.y);
      view.camera.position.copy(this.testCar.position).add(offset);
      view.controls.target.copy(this.testCar.position);
      get("map-panel").scrollIntoView({block: "center"});
    } else if (!follow && this.testCarCamera) {
      const previous = this.testCarCamera;
      this.testCarCamera = null;
      this.app.vehicleManager.vehicleGroup.visible = true;
      if (this.replayTraffic) this.replayTraffic.vehicleGroup.visible = false;
      this.app.roadBuilder.updateSignals(this.app.latestLiveSignals);
      if (view.chaseTarget === this.testCar) {
        view.setCameraView(previous.mode, previous.target);
        view.camera.position.copy(previous.position);
        view.controls.target.copy(previous.lookAt);
        view.controls.update();
      }
    }
    get("route-chase-controls").hidden = !follow;
    get("route-chase-return").hidden = follow || !this.testCar;
  }

  makeTestCar(sample) {
    const car = new THREE.Group();
    for (const [size, y, color] of [[[1.9, .65, 4.2], .65, 0xfbbf24], [[1.55, .62, 2.05], 1.23, 0x28455a]]) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshStandardMaterial({color}));
      mesh.position.y = y;
      car.add(mesh);
    }
    car.position.set(sample[1], 0, sample[2]);
    car.rotation.y = THREE.MathUtils.degToRad(sample[3] || 0);
    this.app.scene3d.scene.add(car);
    this.testCar = car;
  }

  makeRoutePreview() {
    const canvas = get("route-test-map"), context = canvas.getContext("2d");
    const lines = this.testRoute.road_ids.map(id => this.roads.get(id).points);
    const points = lines.flat();
    const xs = points.map(point => point.x), ys = points.map(point => point.y);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const scale = Math.min((canvas.width - 66) / Math.max(1, Math.max(...xs) - Math.min(...xs)),
      (canvas.height - 66) / Math.max(1, Math.max(...ys) - Math.min(...ys)));
    const screen = (x, y) => [canvas.width / 2 + (x - cx) * scale, canvas.height / 2 - (y - cy) * scale];
    const draw = (car) => {
      context.fillStyle = "#081723";
      context.fillRect(0, 0, canvas.width, canvas.height);
      const paths = (roads, color, width) => {
        context.strokeStyle = color;
        context.lineWidth = width;
        context.lineCap = "round";
        context.beginPath();
        for (const line of roads) {
          if (!line.length) continue;
          context.moveTo(...screen(line[0].x, line[0].y));
          for (const point of line.slice(1)) context.lineTo(...screen(point.x, point.y));
        }
        context.stroke();
      };
      paths([...this.roads.values()].map(road => road.points), "#22394b", 5);
      paths([points], "#67e8f9", 5);
      for (const id of this.testRoute.road_ids) {
        const road = this.roads.get(id);
        if (!this.routeJunctions.has(road.to_node)) continue;
        const [x, y] = screen(road.points.at(-1).x, road.points.at(-1).y);
        context.fillStyle = "#d8c9ff";
        context.beginPath();
        context.arc(x, y, 4, 0, 2 * Math.PI);
        context.fill();
        context.font = "11px sans-serif";
        context.fillText(road.to_node.replace("INT_", ""), x + 6, y + 14);
      }
      for (const [point, label] of [[points[0], "Start"], [points.at(-1), "Destination"]]) {
        const [x, y] = screen(point.x, point.y);
        context.fillStyle = "#f8fafc";
        context.font = "13px sans-serif";
        context.fillText(label, x - 14, y - 14);
      }
      if (car) {
        context.fillStyle = "#fbbf24";
        context.beginPath();
        context.arc(...screen(car[1], car[2]), 7, 0, 2 * Math.PI);
        context.fill();
      }
    };
    canvas.hidden = false;
    return draw;
  }

  playTestDrive(report, version) {
    const draw = this.makeRoutePreview(), samples = report.trace;
    if (!samples.length) {
      const wasFollowing = Boolean(this.testCarCamera);
      this.removeTestCar();
      if (wasFollowing) get("route-test-panel").scrollIntoView({block: "nearest"});
      draw(null);
      return Promise.resolve();
    }
    const traffic = report.traffic;
    let trafficFrame = -1;
    if (!this.testCar) this.makeTestCar(samples[0]);
    this.replayTraffic = new VehicleManager(this.app.scene3d.scene);
    this.replayTraffic.vehicleGroup.visible = false;
    this.testCar.visible = get("route-test-panel").open;
    this.setTestCarCamera(get("route-test-panel").open);
    this.testCar.position.set(samples[0][1], 0, samples[0][2]);
    this.testCar.rotation.y = THREE.MathUtils.degToRad(samples[0][3] || 0);
    let cursor = 0;
    let elapsed = samples[0][0], previous = performance.now();
    return new Promise(resolve => {
      const frame = now => {
        if (version !== this.routeTestVersion || !this.testCar) return resolve();
        if (this.testCarCamera && this.app.scene3d.chaseTarget !== this.testCar) {
          this.setTestCarCamera(false);
        }
        const available = samples.at(-1)[0];
        const end = report.finished ? (report.result.seconds ?? report.timeout) : available;
        const speed = report.finished ? Math.max(4, end / 20) : Math.max(1, Math.min(4, available - elapsed));
        elapsed = Math.min(end, elapsed + Math.max(0, now - previous) * speed / 1000);
        previous = now;
        let nextTrafficFrame = trafficFrame;
        while (nextTrafficFrame + 1 < traffic.length && traffic[nextTrafficFrame + 1][0] <= elapsed) nextTrafficFrame++;
        if (nextTrafficFrame !== trafficFrame) {
          trafficFrame = nextTrafficFrame;
          const [, vehicles, signals] = traffic[trafficFrame];
          this.replayTraffic.updateVehicles(vehicles.map(([id, x, y, heading, speed, type, color, road_id]) =>
            ({id, x, y, heading, speed, type, color, road_id, accel: 0})), true);
          this.replaySignals = signals.map(([intersection_id, state_string]) => ({intersection_id, state_string}));
          if (this.testCarCamera) this.app.roadBuilder.updateSignals(this.replaySignals);
        }
        while (cursor + 1 < samples.length && samples[cursor + 1][0] <= elapsed) cursor++;
        const from = samples[cursor], to = samples[cursor + 1] || from;
        const t = from === to ? 0 : Math.min(1, Math.max(0, (elapsed - from[0]) / (to[0] - from[0])));
        const x = THREE.MathUtils.lerp(from[1], to[1], t), y = THREE.MathUtils.lerp(from[2], to[2], t);
        const angle = (from[3] || 0) + ((((to[3] || 0) - (from[3] || 0) + 540) % 360) - 180) * t;
        this.testCar.position.set(x, 0, y);
        this.testCar.rotation.y = THREE.MathUtils.degToRad(angle);
        draw(samples[0][0] <= elapsed ? [elapsed, x, y] : null);
        const nearby = this.replayTraffic?.vehicles.size || 0;
        get("route-chase-time").textContent = `Test car replay · ${elapsed.toFixed(1)} s · ${nearby} vehicles nearby`;
        const controller = report.result?.controller || this.app.hud.currentControllerName || "Selected controller";
        get("route-test-status").textContent = `${controller} · ${elapsed.toFixed(1)} s trip time · ${speed.toFixed(0)}× replay`;
        if (!report.finished || elapsed < end) requestAnimationFrame(frame);
        else {
          const wasFollowing = Boolean(this.testCarCamera);
          this.removeTestCar();
          if (wasFollowing) get("route-test-panel").scrollIntoView({block: "nearest"});
          resolve();
        }
      };
      requestAnimationFrame(frame);
    });
  }

  showTestResults(report) {
    const replay = report.results[0];
    const reference = report.results.find(result => result.id === "PPO_NEIGHBOR") || replay;
    const results = [reference, ...report.results.filter(row => row !== reference)];
    const status = get("route-test-status");
    status.textContent = report.scenario === "uploaded" ? "Single route test · one uploaded schedule"
      : "Single route test · one seed";
    const table = document.createElement("table");
    const header = table.createTHead().insertRow();
    for (const label of ["Controller", "Trip Time", "Difference"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      header.append(cell);
    }
    const body = table.createTBody();
    for (const result of results) {
      const row = body.insertRow();
      row.insertCell().textContent = result.controller;
      row.insertCell().textContent = result.seconds == null ? "Unfinished" : `${result.seconds.toFixed(2)} s`;
      const delta = result.seconds == null || reference.seconds == null ? null : result.seconds - reference.seconds;
      row.insertCell().textContent = result === reference ? "Reference" : delta == null ? "—"
        : delta === 0 ? "Same time" : `${Math.abs(delta).toFixed(2)} s ${delta > 0 ? "longer" : "faster"}`;
    }
    const note = document.createElement("p");
    note.textContent = "All controllers use identical test departures, the same route and the same background traffic. Trip time includes entry waiting.";
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Test Details";
    details.append(summary);
    const conditions = document.createElement("p");
    conditions.textContent = `Empty network at start · Routing off · ${report.background_trips} background departures · ${report.timeout_seconds} s cutoff · ${report.scenario.replaceAll("_", " ")} · ${Object.keys(report.incidents || {}).length} matched restrictions.`;
    details.append(conditions);
    const waits = document.createElement("table");
    const waitHeader = waits.createTHead().insertRow();
    for (const label of ["Controller", "Intersection", "Stopped", "Opposing green", "Clearance", "Other"]) {
      const th = document.createElement("th");
      th.textContent = label;
      th.scope = "col";
      waitHeader.append(th);
    }
    const waitBody = waits.createTBody();
    const seconds = value => `${value.toFixed(2)} s`;
    for (const result of results) {
      if (result.checkpoint) {
        const file = document.createElement("p");
        file.textContent = `${result.controller} · ${result.checkpoint} · deterministic ${result.action_selection}`;
        details.append(file);
      }
      for (const visit of result.intersections || []) {
        const row = waitBody.insertRow();
        const red = visit.seconds.opposing_green || 0, clearance = visit.seconds.clearance || 0;
        for (const value of [result.controller, visit.intersection, seconds(visit.stopped),
          seconds(red), seconds(clearance), seconds(Math.max(0, visit.stopped - red - clearance))]) {
          row.insertCell().textContent = value;
        }
      }
    }
    details.append(waits);
    const exportButton = document.createElement("button");
    exportButton.className = "hud-btn";
    exportButton.type = "button";
    exportButton.textContent = "Download decision and wait log";
    exportButton.addEventListener("click", () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: "application/json"}));
      const link = document.createElement("a");
      link.href = url;
      link.download = "urbanflow-route-test.json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    details.append(exportButton);
    get("route-test-results").replaceChildren(table, note, details);
  }
}
