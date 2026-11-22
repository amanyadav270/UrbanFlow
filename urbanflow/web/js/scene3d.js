/**
 * UrbanFlow AI — 3D Geographic Scene & Camera Manager
 * Implements WebGL rendering, orbit controls, day/night lighting, and tactical camera presets.
 */

export class Scene3D {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.isNightMode = true;
    this.cameraMode = "strategic"; // "strategic", "tactical", "intersection", "chase"
    this.clock = new THREE.Clock();

    this._initScene();
    this._initCamera();
    this._initRenderer();
    this._initLighting();
    this._initControls();
    this._initResizeHandler();

    this.renderCallbacks = [];
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0e17);
    this.scene.fog = new THREE.FogExp2(0x0a0e17, 0.0012);
  }

  _initCamera() {
    const aspect = this.container.clientWidth / Math.max(1, this.container.clientHeight);
    this.camera = new THREE.PerspectiveCamera(45, aspect, 1.5, 3000);
    this.camera.position.set(0, 210, 230);
    this.camera.lookAt(0, 0, 0);
  }

  _initRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.container.appendChild(this.renderer.domElement);
  }

  _initLighting() {
    // Ambient Light
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(this.ambientLight);

    // Hemisphere Sky/Ground Light
    this.hemiLight = new THREE.HemisphereLight(0x94a3b8, 0x1e293b, 0.6);
    this.scene.add(this.hemiLight);

    // Directional Sun Light
    this.dirLight = new THREE.DirectionalLight(0xfffaf0, 1.5);
    this.dirLight.position.set(100, 180, 80);
    this.dirLight.castShadow = true;
    this.dirLight.shadow.mapSize.width = 2048;
    this.dirLight.shadow.mapSize.height = 2048;
    this.dirLight.shadow.camera.near = 10;
    this.dirLight.shadow.camera.far = 800;
    const d = 350;
    this.dirLight.shadow.camera.left = -d;
    this.dirLight.shadow.camera.right = d;
    this.dirLight.shadow.camera.top = d;
    this.dirLight.shadow.camera.bottom = -d;
    this.scene.add(this.dirLight);

    // Default to daylight
    this.setNightMode(true);
  }

  _initControls() {
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05; // don't go below ground
    this.controls.minDistance = 15;
    this.controls.maxDistance = 1500;
    this.controls.target.set(0, 0, 0);
    const canvas = this.renderer.domElement;
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      event.stopImmediatePropagation();
      this._zoomAt(event);
    }, { capture: true, passive: false });
    canvas.addEventListener("dblclick", event => {
      if (this.cameraMode === "chase") return;
      const anchor = this._groundPoint(event);
      if (!anchor) return;
      const offset = this.camera.position.clone().sub(this.controls.target);
      this._smoothCameraTransition(anchor.clone().add(offset.multiplyScalar(0.72)), anchor);
    });
  }

  _groundPoint(event) {
    const bounds = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      2 * (event.clientX - bounds.left) / bounds.width - 1,
      1 - 2 * (event.clientY - bounds.top) / bounds.height
    );
    const ray = new THREE.Raycaster();
    ray.setFromCamera(pointer, this.camera);
    return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), new THREE.Vector3());
  }

  _zoomAt(event) {
    if (this.cameraMode === "chase") return;
    const anchor = this._groundPoint(event) || this.controls.target.clone();
    const distance = this.camera.position.distanceTo(this.controls.target);
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 100 : 1);
    const factor = THREE.MathUtils.clamp(distance * Math.exp(delta * 0.0015),
      this.controls.minDistance, this.controls.maxDistance) / distance;
    this.camera.position.sub(anchor).multiplyScalar(factor).add(anchor);
    this.controls.target.sub(anchor).multiplyScalar(factor).add(anchor);
    this.controls.update();
  }

  zoom(factor) {
    const offset = this.camera.position.clone().sub(this.controls.target);
    offset.setLength(THREE.MathUtils.clamp(offset.length() * factor, this.controls.minDistance, this.controls.maxDistance));
    this.camera.position.copy(this.controls.target).add(offset);
    this.controls.update();
  }

  _initResizeHandler() {
    this.resizeObserver = new ResizeObserver(() => {
      const {clientWidth: width, clientHeight: height} = this.container;
      if (!width || !height) return;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    });
    this.resizeObserver.observe(this.container);
  }

  setNightMode(isNight) {
    this.isNightMode = isNight;
    if (isNight) {
      this.scene.background.setHex(0x0a0e17);
      this.scene.fog.color.setHex(0x0a0e17);
      this.ambientLight.color.setHex(0x83a6d9);
      this.ambientLight.intensity = 0.79;
      this.dirLight.color.setHex(0xb6d4fc);
      this.dirLight.intensity = 0.95;
      this.renderer.toneMappingExposure = 1.08;
    } else {
      this.scene.background.setHex(0x94a3b8);
      this.scene.fog.color.setHex(0x94a3b8);
      this.ambientLight.color.setHex(0xffffff);
      this.ambientLight.intensity = 0.8;
      this.dirLight.color.setHex(0xfffaf0);
      this.dirLight.intensity = 1.6;
      this.renderer.toneMappingExposure = 1.15;
    }
  }

  setNetworkBounds(roads, anchor) {
    const points = roads.flatMap(road => road.points || []);
    if (!points.length) return;
    const xs = points.map(point => point.x);
    const zs = points.map(point => point.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);
    this.networkCenter = new THREE.Vector3((minX + maxX) / 2, 0, (minZ + maxZ) / 2);
    if (Number.isFinite(anchor?.x) && Number.isFinite(anchor?.y)) {
      this.networkCenter.set(anchor.x, 0, anchor.y);
    }
    const {x, z} = this.networkCenter;
    this.networkSize = 2 * Math.max(maxX - x, x - minX, maxZ - z, z - minZ);
    const scale = Math.max(1, this.networkSize / 850);
    this.controls.maxDistance = Math.max(1500, this.networkSize * 3);
    this.camera.far = this.controls.maxDistance * 2;
    this.camera.updateProjectionMatrix();
    this.scene.fog.density = 0.00056 / scale;
    const shadow = this.dirLight.shadow.camera;
    shadow.left = shadow.bottom = -350 * scale;
    shadow.right = shadow.top = 350 * scale;
    shadow.far = 800 * scale;
    shadow.updateProjectionMatrix();
    this.dirLight.target.position.copy(this.networkCenter);
    this.dirLight.position.copy(this.networkCenter).add(new THREE.Vector3(100, 180, 80).multiplyScalar(scale));
    this.scene.add(this.dirLight.target);
    this.setCameraView("tactical");
  }

  setCameraView(mode, targetMesh = null) {
    this.cameraMode = mode;
    this.chaseTarget = targetMesh;
    this.controls.enableRotate = mode !== "tactical";
    this.controls.mouseButtons.LEFT = mode === "tactical" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    this.controls.touches.ONE = mode === "tactical" ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;

    const center = this.networkCenter || new THREE.Vector3();
    const size = this.networkSize || 320;
    if (mode === "strategic" || mode === "reset") {
      this._smoothCameraTransition(center.clone().add(new THREE.Vector3(0, size * 0.69, size * 0.92)), center);
    } else if (mode === "tactical") {
      const height = size * 1.35 / Math.min(1, this.camera.aspect);
      this._smoothCameraTransition(center.clone().add(new THREE.Vector3(0, height, 0.01)), center);
    } else if (mode === "intersection") {
      // Focus on Intersection SW
      this._smoothCameraTransition(new THREE.Vector3(-35, 32, -35), new THREE.Vector3(-60, 2, -60));
    }
  }

  resetCamera() {
    this.setCameraView("strategic");
  }

  focusIntersection(ix, iz) {
    this.cameraMode = "intersection";
    this.controls.enableRotate = true;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    this.controls.touches.ONE = THREE.TOUCH.ROTATE;
    this.chaseTarget = null;
    this._smoothCameraTransition(
      new THREE.Vector3(ix + 57, 97, iz + 62),
      new THREE.Vector3(ix, 2, iz)
    );
  }

  _smoothCameraTransition(targetCamPos, targetLookAt) {
    this.controls.target.copy(targetLookAt);
    this.camera.position.copy(targetCamPos);
    this.controls.update();
  }

  onRender(callback) {
    this.renderCallbacks.push(callback);
  }

  startRenderLoop() {
    const animate = () => {
      requestAnimationFrame(animate);

      const delta = this.clock.getDelta();
      const elapsed = this.clock.getElapsedTime();

      // Follow Camera Mode
      if (this.cameraMode === "chase" && this.chaseTarget) {
        const tPos = this.chaseTarget.position;
        const offset = new THREE.Vector3(0, 5, -12).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.chaseTarget.rotation.y);
        this.camera.position.lerp(tPos.clone().add(offset), 0.1);
        this.controls.target.lerp(tPos, 0.15);
      }

      this.controls.update();

      // Execute custom animation callbacks
      this.renderCallbacks.forEach(cb => cb(delta, elapsed));

      this.renderer.render(this.scene, this.camera);
    };

    animate();
  }
}
