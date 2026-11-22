/**
 * UrbanFlow AI — Main Application Orchestrator
 * Connects the 3D WebGL scene to live backend telemetry and controls.
 */

import { Scene3D } from "./scene3d.js?v=cleaned-upload";
import { RoadBuilder } from "./road_builder.js?v=cleaned-upload";
import { VehicleManager } from "./vehicles.js?v=route-traffic-1";
import { HUDController } from "./hud.js?v=route-highlight-2";
import { Dashboard } from "./dashboard.js?v=route-close-1";

class UrbanFlowApp {
  constructor() {
    this.scene3d = new Scene3D("viewport-container");
    this.roadBuilder = new RoadBuilder(this.scene3d.scene);
    this.vehicleManager = new VehicleManager(this.scene3d.scene);
    this.hud = new HUDController(this);
    this.dashboard = new Dashboard(this);

    this.socket = null;
    this.isConnected = false;
    this.controlsPending = Promise.resolve();
    this.hud.setConnection(false);


    // Raycaster for vehicle picking
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    this._initInteraction();

    this._connectWebSocket();

    // Hook vehicle animation into 60 FPS WebGL render loop
    this.scene3d.onRender((dt, elapsed) => {
      this.vehicleManager.animate(dt, elapsed);
    });

    this.scene3d.startRenderLoop();
  }

  _initInteraction() {
    const canvas = this.scene3d.renderer.domElement;
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      const bounds = canvas.getBoundingClientRect();
      this.mouse.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      this.mouse.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;

      this.raycaster.setFromCamera(this.mouse, this.scene3d.camera);
      const intersects = this.raycaster.intersectObjects(this.vehicleManager.vehicleGroup.children, true);

      if (intersects.length > 0) {
        let hit = intersects[0].object;
        while (hit.parent && !hit.userData.data && hit.parent !== this.vehicleManager.vehicleGroup) {
          hit = hit.parent;
        }

        if (hit.userData && hit.userData.data) {
          const vData = hit.userData.data;
          this.vehicleManager.selectedVehicleId = vData.id;
          this.hud.showVehicleDetails(vData);

          // If chase view is active, follow this vehicle
          if (this.scene3d.cameraMode === "chase") {
            this.scene3d.chaseTarget = hit;
          }
        }
      }
    });
  }

  _connectWebSocket() {
    // Build WebSocket URL from window location
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/live`;

    const statusBadge = document.getElementById("status-badge");
    const statusText = document.getElementById("status-text");

    const connect = () => {
      try {
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
          this.isConnected = true;
          this.hud.setConnection(true);
          if (statusBadge && statusText) {
            statusBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
            statusBadge.style.color = "#10b981";
            statusText.innerText = "Live";
          }
        };

        this.socket.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "topology" && msg.data) {
              this.mapName = msg.data.map_name;
              this.intersections = msg.data.intersections;
              this.focusedIntersectionId = null;
              this.vehicleManager.updateVehicles([]);
              document.getElementById("close-inspector").click();
              this.roadBuilder.buildEnvironment(msg.data);
              this.scene3d.setNetworkBounds(msg.data.roads);
              this.hud.updateCameraSelection();
              this.hud.updateScenarioSelection("");
              if (this.hud) {
                this.hud.updateMapSelection(msg.data.map_name || "classroom_grid");
              }
              this.dashboard.setTopology(msg.data);
            } else if (msg.type === "snapshot" && msg.data) {
              if (msg.data.map_name && msg.data.map_name !== this.mapName) return;
              if (msg.data.demand) this.hud.updateDemand(msg.data.demand, msg.data.demand_revision);
              this.hud.setPlayback(msg.data.paused, msg.data.speed_multiplier, msg.data.timestamp, msg.data.finished, msg.data.effective_speed);
              this._handleSnapshot(
                msg.data,
                msg.intelligence,
                msg.signal_controller || msg.data.signal_controller,
                msg.routing_controller || msg.data.routing_controller
              );
            }
          } catch (e) {
            console.error("Telemetry parse error", e);
          }
        };

        this.socket.onclose = () => {
          this.isConnected = false;
          this.hud.setConnection(false);
          if (statusBadge && statusText) {
            statusBadge.style.borderColor = "rgba(245, 158, 11, 0.4)";
            statusBadge.style.color = "#f59e0b";
            statusText.innerText = "Reconnecting";
          }

          setTimeout(connect, 3000);
        };

        this.socket.onerror = () => {
          this.socket.close();
        };
      } catch (err) {
        setTimeout(connect, 3000);
      }
    };

    connect();
  }

  focusIntersection(id) {
    const junction = this.intersections?.find(item => item.id === id)
      || this.intersections?.find(item => item.has_signals);
    if (junction) {
      this.dashboard.setTestCarCamera(false);
      this.focusedIntersectionId = junction.id;
      this.scene3d.focusIntersection(junction.position.x, junction.position.y);
      this.hud.updateCameraSelection();
    }
  }

  _handleSnapshot(snapshot, intelligence = null, signalController = null, routingController = null) {
    if (snapshot.timestamp < this.lastTimestamp) {
      this.vehicleManager.updateVehicles([]);
      this.scene3d.chaseTarget = null;
    }
    this.lastTimestamp = snapshot.timestamp;
    if (snapshot.vehicles) {
      this.vehicleManager.updateVehicles(snapshot.vehicles, snapshot.speed_multiplier >= 10);
      this.hud.setFleetAvailability(snapshot.vehicles.length > 0);
      const moving = snapshot.vehicles.filter(vehicle => vehicle.speed > 0.5).length;
      document.getElementById("val-moving").textContent = moving;
      this.updateTrafficHold(snapshot, moving);
    }
    if (snapshot.signals) {
      this.latestLiveSignals = snapshot.signals;
      if (!this.dashboard.testCarCamera || !this.dashboard.replaySignals) {
        this.roadBuilder.updateSignals(snapshot.signals);
      }
      this.hud.updateSignals(snapshot.signals);
    }
    if (snapshot.metrics) {
      this.hud.updateMetrics(snapshot.metrics);
    }
    if (intelligence) {
      this.hud.updatePredictiveIntelligence(intelligence);
    }
    this.dashboard.update(snapshot, intelligence);

    const ctrlData = signalController || snapshot.signal_controller || (intelligence ? intelligence.signal_controller : null);
    if (ctrlData && this.hud) {
      this.hud.updateSignalController(ctrlData);
      this.hud.updateAgents(snapshot, ctrlData);
    }

    const routeData = routingController || snapshot.routing_controller;
    if (routeData && this.hud) {
      this.hud.updateRoutingController(routeData);
      if (this.roadBuilder && routeData.storage_warnings) {
        this.roadBuilder.updateStorageWarnings(routeData.storage_warnings);
      }
    }

    // Keep inspector updated if vehicle is selected
    if (this.vehicleManager.selectedVehicleId) {
      const selected = this.vehicleManager.getSelectedVehicle();
      if (selected && selected.data) {
        this.hud.showVehicleDetails(selected.data);
      } else {
        document.getElementById("close-inspector").click();
      }
    }
  }

  updateTrafficHold(snapshot, moving) {
    const notice = document.getElementById("traffic-hold");
    if (snapshot.paused || moving || !snapshot.vehicles.length) {
      this.standstillStart = null;
      notice.hidden = true;
      return;
    }
    if (this.standstillStart == null || snapshot.timestamp < this.standstillStart) {
      this.standstillStart = snapshot.timestamp;
    }
    const seconds = Math.floor(snapshot.timestamp - this.standstillStart);
    notice.hidden = seconds < 5;
    if (seconds < 5) return;
    const clearing = snapshot.signals?.find(signal => signal.phase_name === "All-Red Clearance" && signal.time_in_phase > 5);
    notice.textContent = clearing
      ? `No cars moving for ${seconds} simulated seconds. ${clearing.intersection_id} is holding all-red clearance.`
      : `No cars moving for ${seconds} simulated seconds. Playback is still running.`;
  }

  sendCommand(cmd, successMessage) {
    this.controlsPending = this.controlsPending.then(async () => {
      try {
        const response = await fetch("/api/control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(cmd),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.message || "The setting could not be applied.");
        if (result.controller_info) this.hud.updateSignalController(result.controller_info);
        if (result.routing_info) this.hud.updateRoutingController(result.routing_info);
        if (successMessage) this.hud.showFeedback(successMessage);
        return result;
      } catch (error) {
        this.hud.showFeedback(error.message || "Could not reach the simulator.", true);
        return false;
      }
    });
    return this.controlsPending;
  }

}

// Bootstrap Application
window.addEventListener("DOMContentLoaded", () => {
  try {
    window.urbanFlow = new UrbanFlowApp();
  } catch (error) {
    document.getElementById("status-text").textContent = "3D initialization failed";
    console.error(error);
  }
});
