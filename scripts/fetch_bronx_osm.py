"""Fetches real OpenStreetMap data for a district in the Bronx, New York.
Saves raw OSM JSON data to data/maps/bronx_raw_osm.json.
"""
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Area: Fordham Road & Grand Concourse, Bronx, NY
# Approx 800m x 700m bounding box
BBOX_SOUTH = 40.8580
BBOX_WEST = -73.8950
BBOX_NORTH = 40.8650
BBOX_EAST = -73.8860

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

QUERY = f"""[out:json][timeout:30];
(
  way["highway"~"primary|secondary|tertiary|residential|trunk|service"]({BBOX_SOUTH},{BBOX_WEST},{BBOX_NORTH},{BBOX_EAST});
  way["building"]({BBOX_SOUTH},{BBOX_WEST},{BBOX_NORTH},{BBOX_EAST});
  node["highway"="traffic_signals"]({BBOX_SOUTH},{BBOX_WEST},{BBOX_NORTH},{BBOX_EAST});
);
out body;
>;
out skel qt;"""

def fetch_data():
    print(f"Fetching OSM data for Bronx, NY (bbox: {BBOX_SOUTH},{BBOX_WEST} to {BBOX_NORTH},{BBOX_EAST})...")
    data = urllib.parse.urlencode({"data": QUERY}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_URL,
        data=data,
        headers={"User-Agent": "UrbanFlowAI-Bronx-DigitalTwin/1.0 (urbanflow@traffic.org)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
            res = json.loads(content)
            elements = res.get("elements", [])
            print(f"Successfully fetched {len(elements)} OSM elements!")

            out_dir = Path(__file__).resolve().parent.parent / "data" / "maps"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "bronx_raw_osm.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved raw OSM data to: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
            return True
    except Exception as e:
        print(f"Failed to fetch from Overpass: {e}")
        return False

if __name__ == "__main__":
    success = fetch_data()
    if not success:
        sys.exit(1)
