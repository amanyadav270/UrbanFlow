/**
 * UrbanFlow AI — 3D Geographic Road & Urban Environment Builder
 * Supports both:
 * 1. Real Geographic Digital Twin (e.g. Bronx, NY from OpenStreetMap)
 *    - Real road ribbon polylines with lane widths & markings
 *    - 3D extruded building footprint geometries with realistic heights
 *    - Traffic signal posts at real signalized intersections
 * 2. Synthetic 4-way arterial corridor procedural baseline
 */

export class RoadBuilder {
  constructor(scene) {
    this.scene = scene;
    this.currentGroup = null;
    this.signalBulbs = { ns: [], ew: [] };
    this.signalBulbsByInter = new Map();
    this.currentMapName = "bronx_fordham";
    this.isHeatmapActive = false;
    this.roadGeo = null;
    this.roadVertexMap = new Map();
    this.roadsMap = new Map();
    this.currentRouteHighlight = null;
    this.trafficLayers = [];
    this.nightMode = true;
    this.nightLighting = null;
  }

  clearEnvironment() {
    if (this.currentGroup) {
      this.scene.remove(this.currentGroup);
      this.currentGroup.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
          else child.material.dispose();
        }
        if (child.userData.ownedTexture) child.userData.ownedTexture.dispose();
      });
      this.currentGroup = null;
    }
    this.signalBulbs = { ns: [], ew: [] };
    if (this.signalBulbsByInter) {
      this.signalBulbsByInter.clear();
    }
    this.clearRouteHighlight();
    this.roadsMap.clear();
    this.trafficLayers = [];
    this.nightLighting = null;
  }

  buildEnvironment(topology) {
    this.clearEnvironment();

    const group = new THREE.Group();
    group.name = "UrbanEnvironment";
    this.currentGroup = group;

    const mapName = topology ? (topology.map_name || "bronx_fordham") : "bronx_fordham";
    this.currentMapName = mapName;

    const hasGeographicData = topology && topology.roads && topology.roads.length > 0;

    if (hasGeographicData) {
      this._buildGeographicDigitalTwin(group, topology);
    } else {
      this._buildProceduralCorridor(group);
    }

    this.scene.add(group);
  }

  // =========================================================================
  // REAL GEOGRAPHIC DIGITAL TWIN (BRONX, NY)
  // =========================================================================

  _buildGeographicDigitalTwin(group, topology) {
    // 1. Terrain Ground Plane
    const groundSize = topology.metadata?.ground_size || 1400;
    const cityscape = topology.metadata?.cityscape;
    const groundGeo = new THREE.PlaneGeometry(groundSize * (cityscape ? 1.8 : 1), groundSize * (cityscape ? 1.8 : 1));
    const groundMat = new THREE.MeshStandardMaterial({
      color: cityscape ? 0x252f38 : 0x0f172a,
      roughness: 0.95,
      metalness: 0.05,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    group.add(ground);

    // Subtle geographic orientation grid
    if (!cityscape) {
      const grid = new THREE.GridHelper(groundSize, Math.ceil(groundSize / 20), 0x1e293b, 0x1e293b);
      grid.position.y = 0.01;
      group.add(grid);
    }

    const surfaceColors = { block: 0x334154, park: 0x315843, water: 0x27738b };
    for (const {bounds, kind} of topology.metadata?.surfaces || []) {
      const [x0, y0, x1, y1] = bounds;
      const surface = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0),
        new THREE.MeshStandardMaterial({ color: surfaceColors[kind], roughness: 0.9 }));
      surface.rotation.x = -Math.PI / 2;
      surface.position.set((x0 + x1) / 2, 0.025, (y0 + y1) / 2);
      surface.receiveShadow = true;
      group.add(surface);
    }

    // 2. Real Road Network Ribbon Mesh & Crisp Markings
    const roads = topology.roads || [];
    const managed = new Set(topology.managed_junctions || []);
    this._buildGeographicRoads(group, roads, managed);
    this._buildTrafficRibbons(group, roads);

    // 3. Clean Intersection Junction Slabs & Crosswalks
    const geographic = topology.map_name === "bronx_fordham";
    if (geographic) {
      this._buildRoadAlignedJunctions(group, topology.intersections || [], roads, managed);
    } else {
      this._buildGeographicIntersections(group, topology.intersections || [], topology.metadata);
    }

    // 4. Real 3D Extruded Buildings (strictly outside roads)
    this._buildGeographicBuildings(group, topology.buildings || [], cityscape);
    if (cityscape) this._buildCityDetails(group, topology.metadata.surfaces, topology.buildings);

    // 5. Traffic Signals Positioned Beside Each Approach
    this._buildGeographicSignals(group, topology.intersections || [], geographic ? roads : null);
  }

  _buildGeographicRoads(group, roads, managed) {
    this.roadVertexMap.clear();

    const roadMat = new THREE.MeshStandardMaterial({
      roughness: 0.85,
      metalness: 0.1,
      vertexColors: true,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });

    const whiteMarkingMat = new THREE.MeshBasicMaterial({
      color: 0xf8fafc,
      side: THREE.DoubleSide,
    });

    const yellowMarkingMat = new THREE.MeshBasicMaterial({
      color: 0xfbbf24,
      side: THREE.DoubleSide,
    });

    const roadPositions = [];
    const roadNormals = [];
    const roadColors = [];
    const whiteMarkingPositions = [];
    const yellowMarkingPositions = [];

    const pushQuad = (arr, x0, z0, x1, z1, x2, z2, x3, z3, y) => {
      arr.push(x0, y, z0); arr.push(x1, y, z1); arr.push(x2, y, z2);
      arr.push(x0, y, z0); arr.push(x2, y, z2); arr.push(x3, y, z3);
    };

    roads.forEach((road) => {
      this.roadsMap.set(road.id, road);
      const pts = road.points || [];
      if (pts.length < 2) return;

      const startVertex = roadPositions.length / 3;
      const lanes = Math.max(1, road.lanes || 1);
      const laneWidth = 3.5;
      const roadW = lanes * laneWidth;
      const halfW = roadW / 2.0;

      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i];
        const p1 = pts[i + 1];

        const dx = p1.x - p0.x;
        const dz = p1.y - p0.y; // Y in sim maps to Z in Three.js
        const len = Math.hypot(dx, dz);
        if (len < 0.01) continue;

        const tx = dx / len;
        const tz = dz / len;
        const nx = -tz;
        const nz = tx;

        // 4 Quad Corners for road surface (asphalt at Y = 0.04)
        const yLvl = 0.04;
        const x0L = road.right_edge?.[i].x ?? (p0.x - nx * halfW);
        const z0L = road.right_edge?.[i].y ?? (p0.y - nz * halfW);
        const x0R = road.left_edge?.[i].x ?? (p0.x + nx * halfW);
        const z0R = road.left_edge?.[i].y ?? (p0.y + nz * halfW);

        const x1L = road.right_edge?.[i + 1].x ?? (p1.x - nx * halfW);
        const z1L = road.right_edge?.[i + 1].y ?? (p1.y - nz * halfW);
        const x1R = road.left_edge?.[i + 1].x ?? (p1.x + nx * halfW);
        const z1R = road.left_edge?.[i + 1].y ?? (p1.y + nz * halfW);

        pushQuad(roadPositions, x0L, z0L, x0R, z0R, x1R, z1R, x1L, z1L, yLvl);

        for (let k = 0; k < 6; k++) {
          roadNormals.push(0, 1, 0);
          roadColors.push(0.12, 0.16, 0.23); // Default clean asphalt (#1e293b)
        }

        const mY = 0.055;
        const lineW = 0.15;

        // 1. Solid outer curb line
        const edgeOffset = halfW - lineW * 0.5;
        const rEdgeX0 = p0.x - nx * edgeOffset;
        const rEdgeZ0 = p0.y - nz * edgeOffset;
        const rEdgeX1 = p1.x - nx * edgeOffset;
        const rEdgeZ1 = p1.y - nz * edgeOffset;
        pushQuad(whiteMarkingPositions,
          rEdgeX0 - nx * (lineW * 0.5), rEdgeZ0 - nz * (lineW * 0.5),
          rEdgeX0 + nx * (lineW * 0.5), rEdgeZ0 + nz * (lineW * 0.5),
          rEdgeX1 + nx * (lineW * 0.5), rEdgeZ1 + nz * (lineW * 0.5),
          rEdgeX1 - nx * (lineW * 0.5), rEdgeZ1 - nz * (lineW * 0.5),
          mY
        );

        // 2. Solid yellow median line
        const lEdgeX0 = p0.x + nx * edgeOffset;
        const lEdgeZ0 = p0.y + nz * edgeOffset;
        const lEdgeX1 = p1.x + nx * edgeOffset;
        const lEdgeZ1 = p1.y + nz * edgeOffset;
        pushQuad(yellowMarkingPositions,
          lEdgeX0 - nx * (lineW * 0.5), lEdgeZ0 - nz * (lineW * 0.5),
          lEdgeX0 + nx * (lineW * 0.5), lEdgeZ0 + nz * (lineW * 0.5),
          lEdgeX1 + nx * (lineW * 0.5), lEdgeZ1 + nz * (lineW * 0.5),
          lEdgeX1 - nx * (lineW * 0.5), lEdgeZ1 - nz * (lineW * 0.5),
          mY
        );

        // 3. Dashed White Lane Divider Lines (between multiple lanes of same direction)
        if (lanes > 1) {
          for (let l = 1; l < lanes; l++) {
            const laneOffset = -halfW + l * laneWidth;
            const dashLen = 3.0;
            const gapLen = 3.0;
            const cycle = dashLen + gapLen;
            let curD = 1.0;
            while (curD < len - 1.0) {
              const dEnd = Math.min(len - 0.5, curD + dashLen);
              const t0 = curD / len;
              const t1 = dEnd / len;

              const dx0 = (p0.x + t0 * dx) + nx * laneOffset;
              const dz0 = (p0.y + t0 * dz) + nz * laneOffset;
              const dx1 = (p0.x + t1 * dx) + nx * laneOffset;
              const dz1 = (p0.y + t1 * dz) + nz * laneOffset;

              pushQuad(whiteMarkingPositions,
                dx0 - nx * (lineW * 0.5), dz0 - nz * (lineW * 0.5),
                dx0 + nx * (lineW * 0.5), dz0 + nz * (lineW * 0.5),
                dx1 + nx * (lineW * 0.5), dz1 + nz * (lineW * 0.5),
                dx1 - nx * (lineW * 0.5), dz1 - nz * (lineW * 0.5),
                mY
              );
              curD += cycle;
            }
          }
        }

        // 4. Solid White Stop Line (Stop bar before intersection approach)
        if (i === pts.length - 2 && managed.has(road.to_node)) {
          if (len > 18) {
            for (let lane = 0; lane < lanes; lane++) {
              const offset = (lane - (lanes - 1) / 2) * laneWidth;
              const x = p1.x - tx * 12 + nx * offset;
              const z = p1.y - tz * 12 + nz * offset;
              const arrow = (d, w) => [x + tx * d + nx * w, mY, z + tz * d + nz * w];
              whiteMarkingPositions.push(...arrow(1.3, 0), ...arrow(0, -0.62), ...arrow(0, 0.62));
              whiteMarkingPositions.push(...arrow(0, -0.16), ...arrow(-1.5, -0.16), ...arrow(-1.5, 0.16),
                                         ...arrow(0, -0.16), ...arrow(-1.5, 0.16), ...arrow(0, 0.16));
            }
          }
          const stopThickness = 0.6;
          const stopX0 = p1.x - tx * stopThickness - nx * halfW;
          const stopZ0 = p1.y - tz * stopThickness - nz * halfW;
          const stopX1 = p1.x - tx * stopThickness + nx * halfW;
          const stopZ1 = p1.y - tz * stopThickness + nz * halfW;
          const stopX2 = p1.x + nx * halfW;
          const stopZ2 = p1.y + nz * halfW;
          const stopX3 = p1.x - nx * halfW;
          const stopZ3 = p1.y - nz * halfW;

          pushQuad(whiteMarkingPositions,
            stopX0, stopZ0,
            stopX1, stopZ1,
            stopX2, stopZ2,
            stopX3, stopZ3,
            mY + 0.002
          );
        }
      }

      const endVertex = roadPositions.length / 3;
      this.roadVertexMap.set(road.id, { start: startVertex, count: endVertex - startVertex });
    });

    if (roadPositions.length > 0) {
      const roadGeo = new THREE.BufferGeometry();
      roadGeo.setAttribute("position", new THREE.Float32BufferAttribute(roadPositions, 3));
      roadGeo.setAttribute("normal", new THREE.Float32BufferAttribute(roadNormals, 3));
      roadGeo.setAttribute("color", new THREE.Float32BufferAttribute(roadColors, 3));
      this.roadGeo = roadGeo;

      const roadMesh = new THREE.Mesh(roadGeo, roadMat);
      roadMesh.receiveShadow = true;
      group.add(roadMesh);
    }

    if (whiteMarkingPositions.length > 0) {
      const markGeo = new THREE.BufferGeometry();
      markGeo.setAttribute("position", new THREE.Float32BufferAttribute(whiteMarkingPositions, 3));
      const markMesh = new THREE.Mesh(markGeo, whiteMarkingMat);
      group.add(markMesh);
    }

    if (yellowMarkingPositions.length > 0) {
      const yMarkGeo = new THREE.BufferGeometry();
      yMarkGeo.setAttribute("position", new THREE.Float32BufferAttribute(yellowMarkingPositions, 3));
      const yMarkMesh = new THREE.Mesh(yMarkGeo, yellowMarkingMat);
      group.add(yMarkMesh);
    }
  }

  _buildRoadAlignedJunctions(group, intersections, roads, managed) {
    const asphalt = new THREE.MeshStandardMaterial({ color: 0x161e2e, roughness: 0.85 });
    const white = new THREE.MeshBasicMaterial({ color: 0xf1f5f9, side: THREE.DoubleSide });
    const turn = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const halfHull = points => {
      const chain = [];
      for (const point of points) {
        while (chain.length >= 2 && turn(chain.at(-2), chain.at(-1), point) <= 0) chain.pop();
        chain.push(point);
      }
      return chain.slice(0, -1);
    };

    for (const junction of intersections) {
      if (!managed.has(junction.id)) continue;
      const points = [junction.position];
      for (const road of roads) {
        const inbound = road.to_node === junction.id;
        if (!inbound && road.from_node !== junction.id) continue;
        const index = inbound ? road.points.length - 1 : 0;
        const left = road.left_edge?.[index], right = road.right_edge?.[index];
        if (!left || !right) continue;
        points.push(left, right);
        if (inbound && junction.has_signals) {
          const end = road.points[index], prev = road.points[index - 1];
          const dx = end.x - prev.x, dz = end.y - prev.y;
          const length = Math.hypot(dx, dz);
          if (!length) continue;
          const yaw = Math.atan2(dx, dz);
          for (let across = 0.45; across < road.lanes * 3.5; across += 0.9) {
            const stripe = new THREE.Mesh(new THREE.PlaneGeometry(0.42, 1.5), white);
            stripe.rotation.order = "YXZ";
            stripe.rotation.set(-Math.PI / 2, yaw, 0);
            stripe.position.set(right.x + dz / length * -across + dx / length,
                                0.057, right.y + dx / length * across + dz / length);
            group.add(stripe);
          }
        }
      }
      points.sort((a, b) => a.x - b.x || a.y - b.y);
      const hull = [...halfHull(points), ...halfHull(points.reverse())];
      if (hull.length < 3) continue;
      const shape = new THREE.Shape(hull.map(point => new THREE.Vector2(point.x, -point.y)));
      const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), asphalt);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = 0.041;
      mesh.receiveShadow = true;
      group.add(mesh);
    }
  }

  _buildGeographicIntersections(group, intersections, metadata) {
    if (!intersections || intersections.length === 0) return;
    const junctionRadius = metadata?.junction_half_width || 8;
    const stopLineSetback = metadata?.stop_line_setback || 0;

    const junctionMat = new THREE.MeshStandardMaterial({
      color: 0x161e2e,
      roughness: 0.85,
      metalness: 0.1,
    });

    const crosswalkMat = new THREE.MeshBasicMaterial({
      color: 0xf1f5f9,
      side: THREE.DoubleSide,
    });

    const crosswalkPositions = [];
    const boxMarkings = [];
    const pushQuad = (arr, x0, z0, x1, z1, x2, z2, x3, z3, y) => {
      arr.push(x0, y, z0); arr.push(x1, y, z1); arr.push(x2, y, z2);
      arr.push(x0, y, z0); arr.push(x2, y, z2); arr.push(x3, y, z3);
    };

    intersections.forEach((inter, index) => {
      const pos = inter.position || { x: 0, y: 0 };
      const px = pos.x;
      const pz = pos.y;
      const boxSize = junctionRadius * 2;

      // Clean junction asphalt slab connecting all approaches
      const jGeo = new THREE.PlaneGeometry(boxSize, boxSize);
      const jMesh = new THREE.Mesh(jGeo, junctionMat);
      jMesh.rotation.x = -Math.PI / 2;
      jMesh.position.set(px, 0.041, pz);
      jMesh.receiveShadow = true;
      group.add(jMesh);

      // Zebra crosswalk stripes on all 4 approach entries
      const cwOffset = boxSize / 2 - 1.2;
      const stripeW = 0.4;
      const stripeL = 2.0;
      const mY = 0.056;

      if (stopLineSetback && inter.has_signals) {
        const offset = junctionRadius + stopLineSetback;
        const bar = 0.18;
        pushQuad(crosswalkPositions, px - 7, pz + offset - bar, px, pz + offset - bar,
          px, pz + offset + bar, px - 7, pz + offset + bar, mY);
        pushQuad(crosswalkPositions, px, pz - offset - bar, px + 7, pz - offset - bar,
          px + 7, pz - offset + bar, px, pz - offset + bar, mY);
        pushQuad(crosswalkPositions, px + offset - bar, pz, px + offset + bar, pz,
          px + offset + bar, pz + 7, px + offset - bar, pz + 7, mY);
        pushQuad(crosswalkPositions, px - offset - bar, pz - 7, px - offset + bar, pz - 7,
          px - offset + bar, pz, px - offset - bar, pz, mY);
        if (index % 5 === 1) {
          const edge = 5.5;
          for (const sign of [-1, 1]) {
            const z = pz + sign * edge;
            const x = px + sign * edge;
            pushQuad(boxMarkings, px-edge,z-.12, px+edge,z-.12, px+edge,z+.12, px-edge,z+.12,mY);
            pushQuad(boxMarkings, x-.12,pz-edge, x+.12,pz-edge, x+.12,pz+edge, x-.12,pz+edge,mY);
            for (let gap = 0; gap < 10; gap += 2.5) {
              const ax = px-edge+gap, az = pz-sign*edge, bx = px+edge, bz = pz+sign*(edge-gap);
              pushQuad(boxMarkings, ax,az, ax+.15,az, bx,bz-.15, bx,bz,mY);
              if (gap) pushQuad(boxMarkings, px-edge,az+sign*gap, px-edge+.15,az+sign*gap,
                px+edge-gap+.15,pz+sign*edge, px+edge-gap,pz+sign*edge,mY);
            }
          }
        }
      }

      for (let s = -6.0; s <= 6.0; s += 1.0) {
        // North crosswalk
        pushQuad(crosswalkPositions,
          px + s - stripeW * 0.5, pz + cwOffset - stripeL * 0.5,
          px + s + stripeW * 0.5, pz + cwOffset - stripeL * 0.5,
          px + s + stripeW * 0.5, pz + cwOffset + stripeL * 0.5,
          px + s - stripeW * 0.5, pz + cwOffset + stripeL * 0.5,
          mY
        );
        // South crosswalk
        pushQuad(crosswalkPositions,
          px + s - stripeW * 0.5, pz - cwOffset - stripeL * 0.5,
          px + s + stripeW * 0.5, pz - cwOffset - stripeL * 0.5,
          px + s + stripeW * 0.5, pz - cwOffset + stripeL * 0.5,
          px + s - stripeW * 0.5, pz - cwOffset + stripeL * 0.5,
          mY
        );
        // East crosswalk
        pushQuad(crosswalkPositions,
          px + cwOffset - stripeL * 0.5, pz + s - stripeW * 0.5,
          px + cwOffset + stripeL * 0.5, pz + s - stripeW * 0.5,
          px + cwOffset + stripeL * 0.5, pz + s + stripeW * 0.5,
          px + cwOffset - stripeL * 0.5, pz + s + stripeW * 0.5,
          mY
        );
        // West crosswalk
        pushQuad(crosswalkPositions,
          px - cwOffset - stripeL * 0.5, pz + s - stripeW * 0.5,
          px - cwOffset + stripeL * 0.5, pz + s - stripeW * 0.5,
          px - cwOffset + stripeL * 0.5, pz + s + stripeW * 0.5,
          px - cwOffset - stripeL * 0.5, pz + s + stripeW * 0.5,
          mY
        );
      }
    });

    if (crosswalkPositions.length > 0) {
      const cwGeo = new THREE.BufferGeometry();
      cwGeo.setAttribute("position", new THREE.Float32BufferAttribute(crosswalkPositions, 3));
      const cwMesh = new THREE.Mesh(cwGeo, crosswalkMat);
      group.add(cwMesh);
    }
    if (boxMarkings.length) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(boxMarkings, 3));
      group.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: 0xdba849, side: THREE.DoubleSide})));
    }
  }

  _buildCityDetails(group, surfaces, buildings) {
    const paving = [];
    const glazing = [];
    const treePoints = [];
    const lampPoints = [];
    const rectangle = (x0, z0, x1, z1, y) => {
      paving.push(x0,y,z0, x0,y,z1, x1,y,z1, x0,y,z0, x1,y,z1, x1,y,z0);
    };
    for (const {bounds: [x0,z0,x1,z1], kind} of surfaces) {
      if (kind === "water") {
        rectangle(x0,z1,x1,z1+5,0.075);
        continue;
      }
      rectangle(x0,z0-2,x1,z0+2,0.10);
      rectangle(x0,z1-2,x1,z1+2,0.10);
      rectangle(x0-2,z0+2,x0+2,z1-2,0.10);
      rectangle(x1-2,z0+2,x1+2,z1-2,0.10);
      for (let x = x0 + 12; x < x1 - 8; x += 48) {
        lampPoints.push([x, z0], [x + 4, z1]);
        if (x + 22 < x1 - 7 && kind !== "park") treePoints.push([x + 22, z0 + 3], [x + 15, z1 - 3]);
      }
      for (let z = z0 + 23; z < z1 - 9; z += 55) {
        lampPoints.push([x0, z], [x1, z]);
      }
      if (kind === "park") {
        const midX = (x0+x1)/2, midZ = (z0+z1)/2;
        rectangle(midX-2,z0,midX+2,z1,0.11);
        rectangle(x0,midZ-2,x1,midZ+2,0.11);
        for (let x=x0+12; x<x1-8; x+=19) {
          for (let z=z0+12; z<z1-8; z+=21) {
            if (Math.abs(x-midX)>8 && Math.abs(z-midZ)>8) treePoints.push([x,z]);
          }
        }
      }
    }
    for (const building of buildings) {
      if (!building.city_facade) continue;
      const points = building.polygon;
      for (let i=0; i<points.length; i++) {
        const a=points[i], b=points[(i+1)%points.length];
        const dx=b.x-a.x, dz=b.y-a.y, length=Math.hypot(dx,dz);
        if (length < 5) continue;
        const tx=dx/length, tz=dz/length;
        // Step slightly outside the wall so glazing stays visible without z-fighting.
        const x=a.x+tz*0.04, z=a.y-tx*0.04;
        for (let level=4; level<building.height-2; level+=4) {
          for (let col=2; col<length-3; col+=4) {
            const x0=x+tx*col, z0=z+tz*col, x1=x0+tx*2.6, z1=z0+tz*2.6, top=level+1.75;
            glazing.push(x0,level,z0, x1,level,z1, x1,top,z1, x0,level,z0, x1,top,z1, x0,top,z0);
          }
        }
      }
    }
    for (const [positions, color] of [[paving,0xacaaa0],[glazing,0x3b4c52]]) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions,3));
      const mesh = new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide}));
      group.add(mesh);
    }
    const canopy = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1,1),
      new THREE.MeshStandardMaterial({color:0x477247}),treePoints.length);
    const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.22,0.3,3,5),
      new THREE.MeshStandardMaterial({color:0x665343}),treePoints.length);
    const transform = new THREE.Object3D();
    treePoints.forEach(([x,z],index) => {
      transform.position.set(x,1.5,z);
      transform.scale.set(1,1,1);
      transform.updateMatrix();
      trunk.setMatrixAt(index,transform.matrix);
      transform.position.y=4.1;
      const size=2.3+(index%5)*0.32;
      transform.scale.set(size,size*1.18,size);
      transform.updateMatrix();
      canopy.setMatrixAt(index,transform.matrix);
    });
    canopy.castShadow = true;
    group.add(canopy,trunk);
    this._buildStreetLighting(group, lampPoints);
  }

  setNightMode(night) {
    this.nightMode = night;
    if (this.nightLighting) this.nightLighting.visible = night;
  }

  _buildStreetLighting(group, points) {
    const pole = new THREE.InstancedMesh(new THREE.CylinderGeometry(.12, .17, 5, 5),
      new THREE.MeshStandardMaterial({color: 0x667885}), points.length);
    const lamp = new THREE.InstancedMesh(new THREE.BoxGeometry(.85, .24, .55),
      new THREE.MeshBasicMaterial({color: 0xffd39c, toneMapped: false, fog: false}), points.length);
    const size = 32, pixels = new Uint8Array(size * size * 4);
    for (let i = 0; i < size * size; i++) {
      const distance = Math.hypot(i % size - 15.5, Math.floor(i / size) - 15.5) / 15.5;
      pixels.set([255, 192, 99, Math.round(87 * Math.max(0, 1 - distance) ** 2)], i * 4);
    }
    const texture = new THREE.DataTexture(pixels, size, size, THREE.RGBAFormat);
    texture.needsUpdate = true;
    texture.magFilter = THREE.LinearFilter;
    const poolGeo = new THREE.PlaneGeometry(16, 16).rotateX(-Math.PI / 2);
    const pool = new THREE.InstancedMesh(poolGeo, new THREE.MeshBasicMaterial({map: texture,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false}), points.length);
    pool.userData.ownedTexture = texture;
    const transform = new THREE.Object3D();
    points.forEach(([x, z], index) => {
      transform.position.set(x, 2.6, z);
      transform.updateMatrix(); pole.setMatrixAt(index, transform.matrix);
      transform.position.y = 5.16;
      transform.updateMatrix(); lamp.setMatrixAt(index, transform.matrix);
      transform.position.y = .135;
      transform.updateMatrix(); pool.setMatrixAt(index, transform.matrix);
    });
    this.nightLighting = new THREE.Group();
    this.nightLighting.add(lamp, pool);
    this.nightLighting.visible = this.nightMode;
    group.add(pole, this.nightLighting);
  }

  _buildGeographicBuildings(group, buildings, cityscape) {
    const palette = [
      0x1e293b, // slate
      0x334155, // dark stone
      0x0f172a, // obsidian
      0x1e3a5f, // deep corporate blue
      0x27272a, // dark zinc
      0x1e1b4b, // midnight indigo
      0x3f3f46, // medium graphite
    ];

    const materials = palette.map((color) =>
      new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.4,
        metalness: 0.6,
      })
    );

    const roofMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 }); // subtle cyan rooftop accent

    buildings.forEach((bldg, idx) => {
      const rawPts = bldg.polygon || [];
      if (rawPts.length < 3) return;

      // Filter consecutive identical points and duplicate closing point
      const pts = [];
      for (let i = 0; i < rawPts.length; i++) {
        const cur = rawPts[i];
        if (pts.length === 0 || Math.hypot(cur.x - pts[pts.length - 1].x, cur.y - pts[pts.length - 1].y) > 0.05) {
          pts.push(cur);
        }
      }
      if (pts.length > 2 && Math.hypot(pts[0].x - pts[pts.length - 1].x, pts[0].y - pts[pts.length - 1].y) < 0.05) {
        pts.pop();
      }
      if (pts.length < 3) return;

      const shape = new THREE.Shape();
      shape.moveTo(pts[0].x, -pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        shape.lineTo(pts[i].x, -pts[i].y);
      }
      shape.closePath();

      const height = Math.max(8.0, Math.min(95.0, bldg.height || 16.0));

      try {
        const extrudeSettings = {
          depth: height,
          bevelEnabled: false,
        };
        const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);

        const mat = bldg.color ? new THREE.MeshStandardMaterial({ color: bldg.color, roughness: 0.8 }) : materials[idx % materials.length];
        const mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = -Math.PI / 2;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);

        // Rooftop glowing edge/beacon for architectural night aesthetics
        if (!cityscape && height > 20.0 && idx % 3 === 0) {
          const edgeGeo = new THREE.EdgesGeometry(geo);
          const edgeLine = new THREE.LineSegments(
            edgeGeo,
            new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.4 })
          );
          edgeLine.rotation.x = -Math.PI / 2;
          group.add(edgeLine);
        }
      } catch (err) {
        // Safe skip of degenerate self-intersecting polygons
      }
    });
  }

  _buildGeographicSignals(group, intersections, roads = null, leftHand = false) {
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.3 });
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x090d16, roughness: 0.5 });
    const bulbGeo = new THREE.SphereGeometry(0.24, 16, 16);

    const signalIntersections = intersections.filter((i) => i.has_signals);

    signalIntersections.forEach((inter) => {
      const pos = inter.position || { x: 0, y: 0 };
      const px = pos.x;
      const pz = pos.y; // Sim Y maps to Three.js Z

      const bulbsForInter = { ns: [], ew: [] };

      // Helper to create a single signal post at (postX, postZ) with heading facing oncoming drivers
      const createSignalHead = (postX, postZ, facingAngleY, dirGroup) => {
        const postGroup = new THREE.Group();
        postGroup.position.set(postX, 0, postZ);
        postGroup.rotation.y = facingAngleY;

        // Vertical Pole
        const poleGeo = new THREE.CylinderGeometry(0.12, 0.16, 6.0, 12);
        const pole = new THREE.Mesh(poleGeo, poleMat);
        pole.position.y = 3.0;
        postGroup.add(pole);

        // Mast Arm & Signal Head Box
        const boxGeo = new THREE.BoxGeometry(0.5, 1.4, 0.35);
        const sBox = new THREE.Mesh(boxGeo, boxMat);
        sBox.position.set(0, 5.2, 0);
        postGroup.add(sBox);

        // 3 Bulbs: Red (top), Yellow (mid), Green (bottom) pointing toward local +Z
        const redMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        const redBulb = new THREE.Mesh(bulbGeo, redMat);
        redBulb.position.set(0, 5.65, 0.2);
        postGroup.add(redBulb);

        const yellowMat = new THREE.MeshBasicMaterial({ color: 0x332200 });
        const yellowBulb = new THREE.Mesh(bulbGeo, yellowMat);
        yellowBulb.position.set(0, 5.2, 0.2);
        postGroup.add(yellowBulb);

        const greenMat = new THREE.MeshBasicMaterial({ color: 0x003300 });
        const greenBulb = new THREE.Mesh(bulbGeo, greenMat);
        greenBulb.position.set(0, 4.75, 0.2);
        postGroup.add(greenBulb);

        const bulbSet = { red: redBulb, yellow: yellowBulb, green: greenBulb };
        if (dirGroup === "ns") {
          bulbsForInter.ns.push(bulbSet);
          this.signalBulbs.ns.push(bulbSet);
        } else {
          bulbsForInter.ew.push(bulbSet);
          this.signalBulbs.ew.push(bulbSet);
        }

        group.add(postGroup);
      };

      if (roads) {
        for (const road of roads) {
          if (road.to_node !== inter.id || road.points.length < 2) continue;
          if (leftHand && !road.signal_approach) continue;
          const end = road.points.at(-1), prev = road.points.at(-2), start = road.points[0];
          const dx = end.x - prev.x, dz = end.y - prev.y;
          const length = Math.hypot(dx, dz);
          if (!length) continue;
          const isNS = Math.abs(end.x - start.x) < Math.abs(end.y - start.y);
          const curb = (leftHand ? road.left_edge : road.right_edge).at(-1);
          const side = leftHand ? -1 : 1;
          createSignalHead(curb.x + side * dz / length, curb.y - side * dx / length,
                           Math.atan2(-dx, -dz), road.signal_approach || (isNS ? "ns" : "ew"));
        }
      } else {
        createSignalHead(px - 9.0, pz + 9.0, 0, "ns");
        createSignalHead(px + 9.0, pz - 9.0, Math.PI, "ns");
        createSignalHead(px + 9.0, pz + 9.0, Math.PI / 2, "ew");
        createSignalHead(px - 9.0, pz - 9.0, -Math.PI / 2, "ew");
      }

      this.signalBulbsByInter.set(inter.id, bulbsForInter);
    });
  }

  // =========================================================================
  // PROCEDURAL 4-WAY ARTERIAL CORRIDOR BASELINE
  // =========================================================================

  _buildProceduralCorridor(group) {
    const groundGeo = new THREE.PlaneGeometry(600, 600);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.9,
      metalness: 0.1,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    group.add(ground);

    const grid = new THREE.GridHelper(600, 60, 0x1e293b, 0x1e293b);
    grid.position.y = 0.02;
    group.add(grid);

    const R_LEN = 160.0;
    const ROAD_WIDTH = 14.0;

    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.75,
      metalness: 0.15,
    });

    const ewRoadGeo = new THREE.PlaneGeometry(R_LEN * 2, ROAD_WIDTH);
    const ewRoad = new THREE.Mesh(ewRoadGeo, roadMat);
    ewRoad.rotation.x = -Math.PI / 2;
    ewRoad.position.y = 0.05;
    ewRoad.receiveShadow = true;
    group.add(ewRoad);

    const nsRoadGeo = new THREE.PlaneGeometry(ROAD_WIDTH, R_LEN * 2);
    const nsRoad = new THREE.Mesh(nsRoadGeo, roadMat);
    nsRoad.rotation.x = -Math.PI / 2;
    nsRoad.position.y = 0.051;
    nsRoad.receiveShadow = true;
    group.add(nsRoad);

    this._buildSidewalks(group, R_LEN, ROAD_WIDTH);
    this._buildRoadMarkings(group, R_LEN, ROAD_WIDTH);
    this._buildTrafficSignalGantries(group, ROAD_WIDTH);
    this._buildProceduralBuildings(group, R_LEN, ROAD_WIDTH);
  }

  _buildSidewalks(group, rLen, roadW) {
    const curbMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.8 });
    const curbHeight = 0.2;
    const halfRoad = roadW / 2;
    const cornerSize = rLen - halfRoad;
    const cornerGeo = new THREE.BoxGeometry(cornerSize, curbHeight, cornerSize);

    const corners = [
      { x: -(cornerSize / 2 + halfRoad), z: -(cornerSize / 2 + halfRoad) },
      { x: cornerSize / 2 + halfRoad, z: -(cornerSize / 2 + halfRoad) },
      { x: -(cornerSize / 2 + halfRoad), z: cornerSize / 2 + halfRoad },
      { x: cornerSize / 2 + halfRoad, z: cornerSize / 2 + halfRoad },
    ];

    corners.forEach((c) => {
      const curb = new THREE.Mesh(cornerGeo, curbMat);
      curb.position.set(c.x, curbHeight / 2, c.z);
      curb.receiveShadow = true;
      group.add(curb);
    });
  }

  _buildRoadMarkings(group, rLen, roadW) {
    const markYellow = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
    const halfRoad = roadW / 2;

    const yellowLineGeoH = new THREE.PlaneGeometry(rLen * 2, 0.2);
    const yLineH = new THREE.Mesh(yellowLineGeoH, markYellow);
    yLineH.rotation.x = -Math.PI / 2;
    yLineH.position.set(0, 0.06, 0);
    group.add(yLineH);

    const yellowLineGeoV = new THREE.PlaneGeometry(0.2, rLen * 2);
    const yLineV = new THREE.Mesh(yellowLineGeoV, markYellow);
    yLineV.rotation.x = -Math.PI / 2;
    yLineV.position.set(0, 0.06, 0);
    group.add(yLineV);

    const stopBarMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const stopBarW = halfRoad - 0.5;

    const stopConfigs = [
      { x: -halfRoad - 0.5, z: halfRoad / 2, w: 0.6, h: stopBarW },
      { x: halfRoad + 0.5, z: -halfRoad / 2, w: 0.6, h: stopBarW },
      { x: halfRoad / 2, z: halfRoad + 0.5, w: stopBarW, h: 0.6 },
      { x: -halfRoad / 2, z: -halfRoad - 0.5, w: stopBarW, h: 0.6 },
    ];

    stopConfigs.forEach((sc) => {
      const stop = new THREE.Mesh(new THREE.PlaneGeometry(sc.w, sc.h), stopBarMat);
      stop.rotation.x = -Math.PI / 2;
      stop.position.set(sc.x, 0.065, sc.z);
      group.add(stop);
    });
  }

  _buildTrafficSignalGantries(group, roadW) {
    const halfRoad = roadW / 2;
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.3 });
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x090d16, roughness: 0.5 });

    const corners = [
      { id: "NS_NORTH", pos: [-halfRoad - 1.5, halfRoad + 1.5], rot: 0, dir: "ns" },
      { id: "NS_SOUTH", pos: [halfRoad + 1.5, -halfRoad - 1.5], rot: Math.PI, dir: "ns" },
      { id: "EW_WEST", pos: [-halfRoad - 1.5, -halfRoad - 1.5], rot: Math.PI / 2, dir: "ew" },
      { id: "EW_EAST", pos: [halfRoad + 1.5, halfRoad + 1.5], rot: -Math.PI / 2, dir: "ew" },
    ];

    this.signalBulbs = { ns: [], ew: [] };

    corners.forEach((c) => {
      const postGroup = new THREE.Group();
      postGroup.position.set(c.pos[0], 0, c.pos[1]);
      postGroup.rotation.y = c.rot;

      const poleGeo = new THREE.CylinderGeometry(0.12, 0.15, 6.0, 16);
      const pole = new THREE.Mesh(poleGeo, poleMat);
      pole.position.y = 3.0;
      pole.castShadow = true;
      postGroup.add(pole);

      const armGeo = new THREE.CylinderGeometry(0.09, 0.09, 4.0, 16);
      const arm = new THREE.Mesh(armGeo, poleMat);
      arm.rotation.z = Math.PI / 2;
      arm.position.set(2.0, 5.8, 0);
      postGroup.add(arm);

      const boxGeo = new THREE.BoxGeometry(0.4, 1.2, 0.3);
      const sBox = new THREE.Mesh(boxGeo, boxMat);
      sBox.position.set(3.5, 5.2, 0);
      postGroup.add(sBox);

      const bulbGeo = new THREE.SphereGeometry(0.12, 16, 16);

      const redMat = new THREE.MeshBasicMaterial({ color: 0x220000 });
      const redBulb = new THREE.Mesh(bulbGeo, redMat);
      redBulb.position.set(3.5, 5.55, 0.16);
      postGroup.add(redBulb);

      const yellowMat = new THREE.MeshBasicMaterial({ color: 0x221a00 });
      const yellowBulb = new THREE.Mesh(bulbGeo, yellowMat);
      yellowBulb.position.set(3.5, 5.2, 0.16);
      postGroup.add(yellowBulb);

      const greenMat = new THREE.MeshBasicMaterial({ color: 0x002200 });
      const greenBulb = new THREE.Mesh(bulbGeo, greenMat);
      greenBulb.position.set(3.5, 4.85, 0.16);
      postGroup.add(greenBulb);

      this.signalBulbs[c.dir].push({ red: redBulb, yellow: yellowBulb, green: greenBulb });
      group.add(postGroup);
    });
  }

  _buildProceduralBuildings(group, rLen, roadW) {
    const halfRoad = roadW / 2 + 5.0;
    const buildingPalette = [0x1e293b, 0x334155, 0x0f172a, 0x1e1b4b, 0x172554, 0x042f2e];

    for (let i = 0; i < 28; i++) {
      const w = 14 + Math.random() * 16;
      const d = 14 + Math.random() * 16;
      const h = 20 + Math.random() * 65;

      const bGeo = new THREE.BoxGeometry(w, h, d);
      const bColor = buildingPalette[Math.floor(Math.random() * buildingPalette.length)];
      const bMat = new THREE.MeshStandardMaterial({
        color: bColor,
        roughness: 0.3,
        metalness: 0.7,
      });

      const building = new THREE.Mesh(bGeo, bMat);
      building.castShadow = true;
      building.receiveShadow = true;

      const quadrant = i % 4;
      const distAlong = 30 + Math.floor(i / 4) * 26;
      let posX = 0, posZ = 0;

      if (quadrant === 0) { posX = -halfRoad - w / 2 - 2; posZ = -distAlong; }
      else if (quadrant === 1) { posX = halfRoad + w / 2 + 2; posZ = -distAlong; }
      else if (quadrant === 2) { posX = -distAlong; posZ = halfRoad + d / 2 + 2; }
      else { posX = distAlong; posZ = halfRoad + d / 2 + 2; }

      building.position.set(posX, h / 2, posZ);
      group.add(building);
    }
  }

  // =========================================================================
  // DYNAMIC SIGNAL UPDATE
  // =========================================================================

  updateSignals(signalStates) {
    if (!signalStates || signalStates.length === 0) return;

    // Per-intersection real-time signal update for geographic multi-intersection networks
    if (this.signalBulbsByInter && this.signalBulbsByInter.size > 0) {
      signalStates.forEach((sig) => {
        const interId = sig.intersection_id;
        const bulbs = this.signalBulbsByInter.get(interId);
        if (!bulbs) return;
        const stateStr = sig.state_string || "";
        const isNSGreen = stateStr.includes("NS:G");
        const isNSYellow = stateStr.includes("NS:y");
        const isEWGreen = stateStr.includes("EW:G");
        const isEWYellow = stateStr.includes("EW:y");

        if (bulbs.ns) {
          bulbs.ns.forEach((b) => {
            b.green.material.color.setHex(isNSGreen ? 0x10b981 : 0x003300);
            b.yellow.material.color.setHex(isNSYellow ? 0xf59e0b : 0x332200);
            b.red.material.color.setHex(!isNSGreen && !isNSYellow ? 0xef4444 : 0x330000);
          });
        }
        if (bulbs.ew) {
          bulbs.ew.forEach((b) => {
            b.green.material.color.setHex(isEWGreen ? 0x10b981 : 0x003300);
            b.yellow.material.color.setHex(isEWYellow ? 0xf59e0b : 0x332200);
            b.red.material.color.setHex(!isEWGreen && !isEWYellow ? 0xef4444 : 0x330000);
          });
        }
      });
      return;
    }

    // Procedural single corridor fallback
    if (!this.signalBulbs) return;
    const stateStr = signalStates[0].state_string || "";

    const isNSGreen = stateStr.includes("NS:G");
    const isNSYellow = stateStr.includes("NS:y");
    const isEWGreen = stateStr.includes("EW:G");
    const isEWYellow = stateStr.includes("EW:y");

    if (this.signalBulbs.ns) {
      this.signalBulbs.ns.forEach((b) => {
        b.green.material.color.setHex(isNSGreen ? 0x10b981 : 0x003300);
        b.yellow.material.color.setHex(isNSYellow ? 0xf59e0b : 0x332200);
        b.red.material.color.setHex(!isNSGreen && !isNSYellow ? 0xef4444 : 0x330000);
      });
    }

    if (this.signalBulbs.ew) {
      this.signalBulbs.ew.forEach((b) => {
        b.green.material.color.setHex(isEWGreen ? 0x10b981 : 0x003300);
        b.yellow.material.color.setHex(isEWYellow ? 0xf59e0b : 0x332200);
        b.red.material.color.setHex(!isEWGreen && !isEWYellow ? 0xef4444 : 0x330000);
      });
    }
  }

  // =========================================================================
  // DYNAMIC PREDICTIVE CONGESTION HEATMAP
  // =========================================================================

  setHeatmapMode(active) {
    this.isHeatmapActive = active;
    this.trafficLayers.forEach(({mesh}) => { mesh.visible = active; });
    if (!active) {
      this._resetRoadColors();
    }
  }

  _buildTrafficRibbons(group, roads) {
    for (const [width, opacity, height] of [[2.9, .22, .18], [.78, .90, .22]]) {
      const positions = [], colors = [], spans = new Map();
      const green = new THREE.Color("#45dc8a");
      for (const road of roads) {
        const start = positions.length / 3;
        for (let i = 1; i < road.points.length; i++) {
          const a = road.points[i - 1], b = road.points[i];
          const dx = b.x - a.x, dz = b.y - a.y, length = Math.hypot(dx, dz);
          if (length === 0) continue;
          const x = dz * width / length, z = -dx * width / length;
          positions.push(a.x-x,height,a.y-z, a.x+x,height,a.y+z, b.x+x,height,b.y+z,
            a.x-x,height,a.y-z, b.x+x,height,b.y+z, b.x-x,height,b.y-z);
          for (let vertex = 0; vertex < 6; vertex++) colors.push(green.r, green.g, green.b);
        }
        spans.set(road.id, [start, positions.length / 3]);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({vertexColors: true,
        toneMapped: false, fog: false, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false}));
      mesh.visible = this.isHeatmapActive;
      mesh.renderOrder = height > .2 ? 2 : 1;
      this.trafficLayers.push({mesh, spans});
      group.add(mesh);
    }
  }

  _resetRoadColors() {
    if (!this.roadGeo) return;
    const colorAttr = this.roadGeo.getAttribute("color");
    if (!colorAttr) return;
    const arr = colorAttr.array;
    for (let i = 0; i < arr.length; i += 3) {
      arr[i] = 0.12;
      arr[i + 1] = 0.16;
      arr[i + 2] = 0.23;
    }
    colorAttr.needsUpdate = true;
  }

  updateRoadHeatmap(heatmapData) {
    for (const {mesh, spans} of this.trafficLayers) {
      const colors = mesh.geometry.getAttribute("color");
      for (const [id, info] of Object.entries(heatmapData)) {
        const range = spans.get(id);
        if (!range) continue;
        const color = new THREE.Color(info.color);
        for (let index = range[0]; index < range[1]; index++) colors.setXYZ(index, color.r, color.g, color.b);
      }
      colors.needsUpdate = true;
    }
  }

  highlightRoute(route, originalRoute = null, currentRoadId = null) {
    const key = route?.length ? `${route.join(",")}|${originalRoute?.join(",") || ""}|${currentRoadId || ""}` : null;
    if (key && key === this.routeHighlightKey) return;
    this.clearRouteHighlight();
    if (!route || route.length === 0) return;
    this.routeHighlightKey = key;

    const highlightGroup = new THREE.Group();
    highlightGroup.name = "RouteHighlightGroup";

    const currentIdx = currentRoadId ? route.indexOf(currentRoadId) : 0;
    const remainingRoute = currentIdx >= 0 ? route.slice(currentIdx) : route;

    const getRoutePoints = (roadIds) => {
      const points = [];
      roadIds.forEach((rId) => {
        const road = this.roadsMap.get(rId);
        if (road && road.points && road.points.length > 0) {
          road.points.forEach((p) => {
            // Keep planner guidance visibly above both the road and heatmap.
            points.push(new THREE.Vector3(p.x, 0.82, p.y));
          });
        }
      });
      return points;
    };

    // 1. If rerouted and originalRoute provided, render original remaining path in dashed cyan
    const isRerouted = originalRoute && (originalRoute.length !== route.length || originalRoute.some((r, i) => r !== route[i]));
    if (isRerouted) {
      const origPoints = getRoutePoints(originalRoute);
      if (origPoints.length >= 2) {
        const origGeo = new THREE.BufferGeometry().setFromPoints(origPoints);
        const origMat = new THREE.LineDashedMaterial({
          color: 0x38bdf8,
          linewidth: 2,
          scale: 1,
          dashSize: 8,
          gapSize: 4,
          transparent: true,
          opacity: 0.65,
        });
        const origLine = new THREE.Line(origGeo, origMat);
        origLine.computeLineDistances();
        highlightGroup.add(origLine);
      }
    }

    // 2. Render active/diverted remaining route in glowing vibrant green / cyan line
    const mainPoints = getRoutePoints(remainingRoute);
    if (mainPoints.length >= 2) {
      const mainGeo = new THREE.BufferGeometry().setFromPoints(mainPoints);
      const lineColor = isRerouted ? 0x10b981 : 0x06b6d4; // bright emerald for diverted, cyan for original
      const strip = [];
      for (let i = 1; i < mainPoints.length; i++) {
        const a = mainPoints[i - 1], b = mainPoints[i];
        const dx = b.x - a.x, dz = b.z - a.z, length = Math.hypot(dx, dz);
        if (!length) continue;
        const x = dz * 1.9 / length, z = -dx * 1.9 / length;
        strip.push(a.x-x,a.y,a.z-z, a.x+x,a.y,a.z+z, b.x+x,b.y,b.z+z,
          a.x-x,a.y,a.z-z, b.x+x,b.y,b.z+z, b.x-x,b.y,b.z-z);
      }
      const routeGeometry = new THREE.BufferGeometry();
      routeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(strip, 3));
      const routeStrip = new THREE.Mesh(routeGeometry, new THREE.MeshBasicMaterial({color: lineColor,
        fog: false, toneMapped: false, transparent: true, opacity: .80, side: THREE.DoubleSide, depthWrite: false, depthTest: false}));
      routeStrip.renderOrder = 12;
      highlightGroup.add(routeStrip);
      const mainMat = new THREE.LineBasicMaterial({
        color: lineColor,
        linewidth: 4,
        transparent: true,
        opacity: 0.95,
      });
      const mainLine = new THREE.Line(mainGeo, mainMat);
      mainLine.renderOrder = 13;
      highlightGroup.add(mainLine);

      // Destination waypoint marker (small glowing diamond/cylinder at destination point)
      const lastPt = mainPoints[mainPoints.length - 1];
      const markerGeo = new THREE.CylinderGeometry(1.5, 0.1, 4.0, 8);
      const markerMat = new THREE.MeshBasicMaterial({
        color: lineColor,
        transparent: true,
        opacity: 0.85,
      });
      const marker = new THREE.Mesh(markerGeo, markerMat);
      marker.position.copy(lastPt);
      marker.position.y += 2.0;
      highlightGroup.add(marker);
    }

    this.scene.add(highlightGroup);
    this.currentRouteHighlight = highlightGroup;
  }

  clearRouteHighlight() {
    this.routeHighlightKey = null;
    if (this.currentRouteHighlight) {
      this.scene.remove(this.currentRouteHighlight);
      this.currentRouteHighlight.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      this.currentRouteHighlight = null;
    }
  }

  updateStorageWarnings(warnings) {
    // Only apply diagnostic color overlays if heatmap/diagnostics is explicitly enabled by the user
    if (!this.isHeatmapActive || !warnings || warnings.length === 0 || !this.roadGeo || !this.roadVertexMap) return;

    const colorAttr = this.roadGeo.getAttribute("color");
    if (!colorAttr) return;
    const arr = colorAttr.array;

    warnings.forEach((w) => {
      const range = this.roadVertexMap.get(w.road_id);
      if (!range) return;
      // High amber / red warning tint for roads with heavy capacity reservations
      const c = new THREE.Color(w.reserved_vehicles > 2 ? 0xef4444 : 0xf59e0b);
      for (let i = range.start; i < range.start + range.count; i++) {
        arr[i * 3] = c.r;
        arr[i * 3 + 1] = c.g;
        arr[i * 3 + 2] = c.b;
      }
    });
    colorAttr.needsUpdate = true;
  }
}
