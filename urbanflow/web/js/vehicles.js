/**
 * UrbanFlow AI — 3D Vehicle Manager
 * Renders diverse 3D vehicles (sedans, vans, transit buses, emergency vehicles)
 * with functional headlights, brake lights, emergency strobes, and 60 FPS smooth interpolation.
 */

export class VehicleManager {
  constructor(scene) {
    this.scene = scene;
    this.vehicles = new Map(); // id -> { mesh, targetPos, targetHeading, prevPos, prevHeading, ... }
    this.selectedVehicleId = null;
    this.vehicleGroup = new THREE.Group();
    this.vehicleGroup.name = "VehiclesGroup";
    this.scene.add(this.vehicleGroup);

    // Shared reusable materials & geometries for peak performance on Intel Iris Xe
    this._initSharedAssets();
  }

  _initSharedAssets() {
    this.wheelGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.28, 12);
    this.wheelMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    this.glassMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      metalness: 0.9,
      roughness: 0.1,
      transparent: true,
      opacity: 0.75,
    });
    this.headlightMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.brakelightMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
  }

  _createVehicleMesh(vData) {
    const group = new THREE.Group();
    group.name = `veh_${vData.id}`;
    group.userData = { id: vData.id, type: vData.type, data: vData };

    const colorHex = parseInt((vData.color || "#2563eb").replace("#", "0x"), 16);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: colorHex,
      metalness: 0.5,
      roughness: 0.35,
    });

    let length = 4.2, width = 1.9, height = 1.35;
    if (vData.type === "van") { length = 5.4; width = 2.1; height = 2.2; }
    else if (vData.type === "bus") { length = 10.5; width = 2.6; height = 3.1; }
    else if (vData.type === "emergency") { length = 5.0; width = 2.0; height = 1.65; }

    // 1. Lower Chassis
    const chassisGeo = new THREE.BoxGeometry(width, height * 0.5, length);
    const chassis = new THREE.Mesh(chassisGeo, bodyMat);
    chassis.position.y = height * 0.25 + 0.35;
    chassis.castShadow = true;
    chassis.receiveShadow = true;
    group.add(chassis);

    // 2. Cabin / Greenhouse
    const cabinLen = length * (vData.type === "bus" ? 0.92 : (vData.type === "van" ? 0.65 : 0.52));
    const cabinH = height * 0.55;
    const cabinGeo = new THREE.BoxGeometry(width * 0.88, cabinH, cabinLen);
    const cabin = new THREE.Mesh(cabinGeo, vData.type === "bus" ? this.glassMat : bodyMat);
    cabin.position.set(0, height * 0.75 + 0.35, vData.type === "sedan" ? -0.2 : 0);
    cabin.castShadow = true;
    group.add(cabin);

    // Windshield (for non-bus)
    if (vData.type !== "bus") {
      const windGeo = new THREE.BoxGeometry(width * 0.85, cabinH * 0.7, cabinLen * 0.95);
      const wind = new THREE.Mesh(windGeo, this.glassMat);
      wind.position.copy(cabin.position);
      group.add(wind);
    }

    // 3. Four Wheels
    const halfW = width / 2 + 0.05;
    const halfL = length * 0.32;
    const wheelY = 0.35;

    const wheelPositions = [
      [-halfW, wheelY, halfL],
      [halfW, wheelY, halfL],
      [-halfW, wheelY, -halfL],
      [halfW, wheelY, -halfL],
    ];

    wheelPositions.forEach(p => {
      const wheel = new THREE.Mesh(this.wheelGeo, this.wheelMat);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(p[0], p[1], p[2]);
      wheel.castShadow = true;
      group.add(wheel);
    });

    // 4. Headlights (at front, which is +Z in vehicle local space)
    const hlGeo = new THREE.BoxGeometry(0.35, 0.18, 0.1);
    const hlLeft = new THREE.Mesh(hlGeo, this.headlightMat);
    hlLeft.position.set(-width * 0.35, height * 0.35, length / 2 + 0.05);
    group.add(hlLeft);

    const hlRight = new THREE.Mesh(hlGeo, this.headlightMat);
    hlRight.position.set(width * 0.35, height * 0.35, length / 2 + 0.05);
    group.add(hlRight);

    // 5. Brake Lights (at back, which is -Z in vehicle local space)
    const blGeo = new THREE.BoxGeometry(0.35, 0.16, 0.1);
    const blLeft = new THREE.Mesh(blGeo, this.brakelightMat);
    blLeft.position.set(-width * 0.35, height * 0.35, -length / 2 - 0.05);
    group.add(blLeft);

    const blRight = new THREE.Mesh(blGeo, this.brakelightMat);
    blRight.position.set(width * 0.35, height * 0.35, -length / 2 - 0.05);
    group.add(blRight);

    // 6. Emergency Flasher Strobe on Roof
    if (vData.type === "emergency") {
      const strobeGroup = new THREE.Group();
      strobeGroup.name = "EmergencyStrobe";
      strobeGroup.position.set(0, height + 0.45, 0);

      const sGeo = new THREE.BoxGeometry(0.8, 0.2, 0.3);
      const sLeft = new THREE.Mesh(sGeo, new THREE.MeshBasicMaterial({ color: 0x3b82f6 }));
      sLeft.position.x = -0.4;
      strobeGroup.add(sLeft);

      const sRight = new THREE.Mesh(sGeo, new THREE.MeshBasicMaterial({ color: 0xef4444 }));
      sRight.position.x = 0.4;
      strobeGroup.add(sRight);

      group.add(strobeGroup);
      group.userData.strobe = strobeGroup;
    }

    return group;
  }

  updateVehicles(vList, fastForward = false) {
    const currentIds = new Set();

    vList.forEach(vData => {
      currentIds.add(vData.id);
      let record = this.vehicles.get(vData.id);

      if (!record) {
        // Create new vehicle
        const mesh = this._createVehicleMesh(vData);
        mesh.position.set(vData.x, 0, vData.y); // Note: world X, Z in Three.js corresponds to X, Y in simulation
        // Authoritative rotation: local +Z (headlights) aligns with travel direction
        mesh.rotation.y = THREE.MathUtils.degToRad(vData.heading);
        mesh.userData.data = vData;
        this.vehicleGroup.add(mesh);

        record = {
          id: vData.id,
          mesh: mesh,
          currentPos: new THREE.Vector3(vData.x, 0, vData.y),
          targetPos: new THREE.Vector3(vData.x, 0, vData.y),
          currentHeading: vData.heading,
          targetHeading: vData.heading,
          speed: vData.speed,
          accel: vData.accel,
          data: vData,
        };
        this.vehicles.set(vData.id, record);
      } else {
        // Update target positions for smooth interpolation
        record.targetPos.set(vData.x, 0, vData.y);
        record.targetHeading = vData.heading;
        if (fastForward && (record.data.road_id !== vData.road_id || vData.road_id.startsWith("CONN__"))) {
          // Smooth motion along the same road while keeping turns on their reported paths.
          record.currentPos.copy(record.targetPos);
          record.currentHeading = record.targetHeading;
        }
        record.speed = vData.speed;
        record.accel = vData.accel;
        record.data = vData;
        record.mesh.userData.data = vData;
      }
    });

    // Remove departed vehicles
    for (const [vId, record] of this.vehicles.entries()) {
      if (!currentIds.has(vId)) {
        this.vehicleGroup.remove(record.mesh);
        const geometries = new Set();
        const materials = new Set();
        record.mesh.traverse(child => {
          if (!child.isMesh) return;
          if (child.geometry !== this.wheelGeo) geometries.add(child.geometry);
          if (![this.wheelMat, this.glassMat, this.headlightMat, this.brakelightMat].includes(child.material)) {
            materials.add(child.material);
          }
        });
        geometries.forEach(geometry => geometry.dispose());
        materials.forEach(material => material.dispose());
        this.vehicles.delete(vId);
      }
    }
  }

  /**
   * Called every frame in 60 FPS animation loop for buttery smooth interpolation.
   */
  dispose() {
    this.updateVehicles([]);
    this.scene.remove(this.vehicleGroup);
    for (const asset of [this.wheelGeo, this.wheelMat, this.glassMat, this.headlightMat, this.brakelightMat]) {
      asset.dispose();
    }
  }

  animate(deltaTime, elapsedTime) {
    const LERP_FACTOR = 1 - Math.exp(-15 * deltaTime);

    this.vehicles.forEach(record => {
      // Position lerp
      record.currentPos.lerp(record.targetPos, LERP_FACTOR);
      record.mesh.position.x = record.currentPos.x;
      record.mesh.position.z = record.currentPos.z;

      // Heading lerp with angle wrap-around
      let diff = (record.targetHeading - record.currentHeading) % 360;
      if (diff > 180) diff -= 360;
      if (diff < -180) diff += 360;
      record.currentHeading += diff * LERP_FACTOR;
      // Authoritative rotation: local +Z (headlights) aligns with travel direction
      record.mesh.rotation.y = THREE.MathUtils.degToRad(record.currentHeading);

      // Emergency Strobe Flash
      if (record.mesh.userData.strobe) {
        const flash = Math.sin(elapsedTime * 18.0) > 0;
        record.mesh.userData.strobe.visible = flash;
      }
    });
  }

  getSelectedVehicle() {
    if (this.selectedVehicleId && this.vehicles.has(this.selectedVehicleId)) {
      return this.vehicles.get(this.selectedVehicleId);
    }
    return null;
  }
}
