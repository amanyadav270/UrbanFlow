"""OpenStreetMap (OSM) Ingestion and Topological Processor for UrbanFlow AI.

Parses OSM XML data, projects WGS84 geographic coordinates into local metric Cartesian space,
extracts road networks with lane and directional rules, identifies intersections and traffic signals,
and extrudes 3D building footprint geometries with heights.
"""
from __future__ import annotations

import math
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set


# Earth radius in meters (WGS84)
EARTH_RADIUS = 6378137.0


def geo_to_local(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Converts WGS84 (lat, lon) to local Euclidean (x, y) in meters relative to (lat0, lon0)."""
    rad = math.pi / 180.0
    lat_r = lat * rad
    lon_r = lon * rad
    lat0_r = lat0 * rad
    lon0_r = lon0 * rad

    x = (lon_r - lon0_r) * math.cos(lat0_r) * EARTH_RADIUS
    y = (lat_r - lat0_r) * EARTH_RADIUS
    return (round(x, 2), round(y, 2))


def local_to_geo(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Inverse projection from local (x, y) meters back to WGS84 (lat, lon)."""
    rad = math.pi / 180.0
    lat0_r = lat0 * rad

    lat = lat0 + (y / EARTH_RADIUS) / rad
    lon = lon0 + (x / (EARTH_RADIUS * math.cos(lat0_r))) / rad
    return (round(lat, 6), round(lon, 6))


class OSMImporter:
    """Imports, parses, and normalizes OSM XML into UrbanFlow Digital Twin format."""

    VEHICULAR_HIGHWAYS = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "residential", "unclassified", "service", "living_street",
        "motorway_link", "trunk_link", "primary_link", "secondary_link",
    }

    DEFAULT_LANES = {
        "motorway": 3,
        "trunk": 3,
        "primary": 2,
        "secondary": 2,
        "tertiary": 2,
        "residential": 1,
        "service": 1,
        "unclassified": 1,
    }

    DEFAULT_SPEEDS = {
        "primary": 13.4,     # ~30 mph in m/s
        "secondary": 11.2,   # ~25 mph (NYC city speed limit)
        "tertiary": 11.2,
        "residential": 8.9,  # ~20 mph
        "service": 5.5,
    }

    def __init__(self, osm_file_path: Path):
        self.file_path = Path(osm_file_path)
        self.nodes: Dict[str, Tuple[float, float]] = {}  # id -> (lat, lon)
        self.node_tags: Dict[str, Dict[str, str]] = {}
        self.highways: List[Dict[str, Any]] = []
        self.buildings: List[Dict[str, Any]] = []
        self.center_lat = 0.0
        self.center_lon = 0.0

    def parse(self, root=None) -> Dict[str, Any]:
        """Parses the OSM XML file and returns structured geographic digital twin topology."""
        if root is None:
            root = ET.parse(self.file_path).getroot()

        # 1. Parse Nodes
        lats, lons = [], []
        for n in root.findall("node"):
            n_id = n.get("id")
            lat = float(n.get("lat"))
            lon = float(n.get("lon"))
            self.nodes[n_id] = (lat, lon)
            lats.append(lat)
            lons.append(lon)

            tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
            if tags:
                self.node_tags[n_id] = tags

        if not lats or not lons:
            raise ValueError("No nodes found in OSM dataset")

        self.center_lat = sum(lats) / len(lats)
        self.center_lon = sum(lons) / len(lons)

        # 2. Parse Ways
        for w in root.findall("way"):
            w_id = w.get("id")
            tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
            nd_refs = [nd.get("ref") for nd in w.findall("nd") if nd.get("ref") in self.nodes]

            if "highway" in tags and tags["highway"] in self.VEHICULAR_HIGHWAYS:
                if len(nd_refs) >= 2:
                    self.highways.append({
                        "id": w_id,
                        "nodes": nd_refs,
                        "tags": tags,
                    })

            if "building" in tags and len(nd_refs) >= 3:
                self.buildings.append({
                    "id": w_id,
                    "nodes": nd_refs,
                    "tags": tags,
                })

        # 3. Build Processed Road Network and Intersections
        intersections, roads = self._process_road_network()

        # 4. Process 3D Building Footprints
        building_polygons = self._process_buildings()

        return {
            "metadata": {
                "source": "OpenStreetMap",
                "center_lat": round(self.center_lat, 6),
                "center_lon": round(self.center_lon, 6),
                "bounds": {
                    "min_lat": min(lats),
                    "max_lat": max(lats),
                    "min_lon": min(lons),
                    "max_lon": max(lons),
                },
                "total_roads": len(roads),
                "total_intersections": len(intersections),
                "total_buildings": len(building_polygons),
            },
            "intersections": intersections,
            "roads": roads,
            "buildings": building_polygons,
        }

    def _process_road_network(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Segments ways into connected road segments and discovers topological intersections."""
        # Count node occurrences across all highways to find true intersection vertices
        node_usage: Dict[str, int] = {}
        for h in self.highways:
            for n_id in h["nodes"]:
                node_usage[n_id] = node_usage.get(n_id, 0) + 1

        intersections_dict: Dict[str, Dict[str, Any]] = {}
        roads: List[Dict[str, Any]] = []

        road_counter = 0

        for h in self.highways:
            tags = h["tags"]
            highway_type = tags.get("highway", "residential")
            name = tags.get("name", f"Street_{h['id']}")
            try:
                lanes = max(1, int(tags.get("lanes", self.DEFAULT_LANES.get(highway_type, 1))))
            except (ValueError, TypeError):
                lanes = self.DEFAULT_LANES.get(highway_type, 1)
            speed_limit = self._parse_speed(tags.get("maxspeed"), default=self.DEFAULT_SPEEDS.get(highway_type, 11.2))

            oneway = tags.get("oneway", "yes" if tags.get("junction") == "roundabout" else "no").lower() in ("yes", "1", "true")
            oneway_reverse = tags.get("oneway", "no") == "-1"

            # Split way into road segments at intersection nodes
            curr_segment_nodes = [h["nodes"][0]]

            for i in range(1, len(h["nodes"])):
                n_id = h["nodes"][i]
                curr_segment_nodes.append(n_id)

                is_end = (i == len(h["nodes"]) - 1)
                is_junc = (node_usage.get(n_id, 0) > 1 or self.node_tags.get(n_id, {}).get("highway") == "traffic_signals")

                if is_junc or is_end:
                    start_node = curr_segment_nodes[0]
                    end_node = curr_segment_nodes[-1]

                    if start_node != end_node and len(curr_segment_nodes) >= 2:
                        road_counter += 1
                        points_local = [
                            geo_to_local(self.nodes[nid][0], self.nodes[nid][1], self.center_lat, self.center_lon)
                            for nid in curr_segment_nodes
                        ]
                        points_geo = [
                            {"lat": self.nodes[nid][0], "lon": self.nodes[nid][1]}
                            for nid in curr_segment_nodes
                        ]

                        seg_len = sum(
                            math.hypot(points_local[k+1][0] - points_local[k][0], points_local[k+1][1] - points_local[k][1])
                            for k in range(len(points_local) - 1)
                        )

                        # Register intersection nodes
                        for int_nid in (start_node, end_node):
                            if int_nid not in intersections_dict:
                                int_lat, int_lon = self.nodes[int_nid]
                                int_x, int_y = geo_to_local(int_lat, int_lon, self.center_lat, self.center_lon)
                                n_tags = self.node_tags.get(int_nid, {})
                                has_signal = (
                                    n_tags.get("highway") == "traffic_signals"
                                    or highway_type in ("primary", "secondary") and node_usage.get(int_nid, 0) >= 3
                                )
                                intersections_dict[int_nid] = {
                                    "id": f"int_{int_nid}",
                                    "osm_id": int_nid,
                                    "position": {"x": int_x, "y": int_y, "z": 0.0},
                                    "geo": {"lat": int_lat, "lon": int_lon},
                                    "has_signals": has_signal,
                                    "connected_roads": [],
                                }

                        # Forward Road Segment
                        if not oneway_reverse:
                            fwd_id = f"road_{h['id']}_{road_counter}_fwd"
                            roads.append({
                                "id": fwd_id,
                                "osm_way_id": h["id"],
                                "name": name,
                                "type": highway_type,
                                "from_node": f"int_{start_node}",
                                "to_node": f"int_{end_node}",
                                "lanes": lanes,
                                "length": round(seg_len, 2),
                                "speed_limit": round(speed_limit, 2),
                                "points": [{"x": p[0], "y": p[1], "z": 0.0} for p in points_local],
                                "geo_points": points_geo,
                                "oneway": oneway,
                            })
                            intersections_dict[start_node]["connected_roads"].append(fwd_id)
                            intersections_dict[end_node]["connected_roads"].append(fwd_id)

                        # Reverse Road Segment (if bidirectional)
                        if not oneway:
                            road_counter += 1
                            rev_id = f"road_{h['id']}_{road_counter}_rev"
                            roads.append({
                                "id": rev_id,
                                "osm_way_id": h["id"],
                                "name": name,
                                "type": highway_type,
                                "from_node": f"int_{end_node}",
                                "to_node": f"int_{start_node}",
                                "lanes": lanes,
                                "length": round(seg_len, 2),
                                "speed_limit": round(speed_limit, 2),
                                "points": [{"x": p[0], "y": p[1], "z": 0.0} for p in reversed(points_local)],
                                "geo_points": list(reversed(points_geo)),
                                "oneway": oneway,
                            })
                            intersections_dict[end_node]["connected_roads"].append(rev_id)
                            intersections_dict[start_node]["connected_roads"].append(rev_id)

                    curr_segment_nodes = [n_id]

        intersections = list(intersections_dict.values())
        return intersections, roads

    def _process_buildings(self) -> List[Dict[str, Any]]:
        """Processes building footprints and calculates realistic 3D building heights."""
        processed = []
        for b in self.buildings:
            tags = b["tags"]
            b_id = b["id"]

            points_local = [
                geo_to_local(self.nodes[nid][0], self.nodes[nid][1], self.center_lat, self.center_lon)
                for nid in b["nodes"]
            ]

            if len(points_local) < 3:
                continue

            # Ensure polygon is closed
            if points_local[0] != points_local[-1]:
                points_local.append(points_local[0])

            # Calculate footprint area using shoelace formula
            area = 0.5 * abs(sum(
                points_local[k][0] * points_local[k+1][1] - points_local[k+1][0] * points_local[k][1]
                for k in range(len(points_local) - 1)
            ))

            if area < 25.0:  # filter tiny artifacts < 25 sqm
                continue

            # Height Extraction & Heuristic Estimation
            height = self._extract_building_height(tags, area, b_id)

            processed.append({
                "id": f"bldg_{b_id}",
                "name": tags.get("name", tags.get("addr:housenumber", "Building")),
                "type": tags.get("building", "yes"),
                "height": round(height, 1),
                "levels": tags.get("building:levels", "N/A"),
                "area_sqm": round(area, 1),
                "polygon": [{"x": p[0], "y": p[1]} for p in points_local],
            })

        return processed

    def _extract_building_height(self, tags: Dict[str, str], area: float, b_id: str) -> float:
        """Determines height from tags or estimates realistic urban height for Bronx blocks."""
        if "height" in tags:
            try:
                raw_h = tags["height"].replace("m", "").replace("meters", "").strip()
                if "'" in raw_h:
                    return float(raw_h.replace("'", "")) * 0.3048
                return float(raw_h)
            except Exception:
                pass

        if "building:levels" in tags:
            try:
                levels = float(tags["building:levels"])
                return max(7.0, levels * 3.5)
            except Exception:
                pass

        # Realistic Bronx urban height distribution:
        # Typical residential/commercial blocks along Grand Concourse / Fordham are 4-8 stories (14m - 28m)
        pseudo_seed = int(b_id) % 100
        if area > 1200.0:  # large institutional/commercial complex
            return 24.0 + (pseudo_seed % 18)
        elif area > 500.0:  # mid-rise apartment building
            return 16.0 + (pseudo_seed % 14)
        else:  # smaller low-rise commercial/residential
            return 9.0 + (pseudo_seed % 8)

    def _parse_speed(self, val: Optional[str], default: float) -> float:
        if not val:
            return default
        try:
            clean = val.lower().replace("mph", "").replace("km/h", "").strip()
            num = float(clean)
            if "mph" in val.lower() or num <= 65:  # likely mph in US
                return num * 0.44704
            return num / 3.6  # km/h
        except Exception:
            return default


def convert_and_save_bronx_map(
    osm_path: Path,
    out_json_path: Path
) -> Dict[str, Any]:
    """Reads OSM file, parses topology, and saves self-contained JSON digital twin map."""
    importer = OSMImporter(osm_path)
    data = importer.parse()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    osm_file = Path("D:/TRAFFIC/urbanflow/data/maps/bronx_fordham.osm")
    out_file = Path("D:/TRAFFIC/urbanflow/data/maps/bronx_fordham.json")
    print(f"Converting {osm_file} -> {out_file}...")
    result = convert_and_save_bronx_map(osm_file, out_file)
    meta = result["metadata"]
    print("Conversion Complete!")
    print(f"  Center: Lat {meta['center_lat']}, Lon {meta['center_lon']}")
    print(f"  Roads: {meta['total_roads']}")
    print(f"  Intersections: {meta['total_intersections']}")
    print(f"  Buildings: {meta['total_buildings']}")
    print(f"  Saved to: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
