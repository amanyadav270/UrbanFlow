"""Unit tests for the OpenStreetMap (OSM) ingestion and geographic processing pipeline."""
import pytest
from pathlib import Path
from urbanflow.core.osm_importer import OSMImporter, geo_to_local, local_to_geo


def test_projection_roundtrip():
    lat0, lon0 = 40.861582, -73.890306
    test_lat, test_lon = 40.862406, -73.891908

    x, y = geo_to_local(test_lat, test_lon, lat0, lon0)
    assert abs(x) > 0.0
    assert abs(y) > 0.0

    r_lat, r_lon = local_to_geo(x, y, lat0, lon0)
    assert abs(r_lat - test_lat) < 1e-5
    assert abs(r_lon - test_lon) < 1e-5


def test_osm_importer_parses_bronx_file():
    osm_path = Path(__file__).resolve().parent.parent / "data" / "maps" / "bronx_fordham.osm"
    assert osm_path.exists(), f"OSM data file missing: {osm_path}"

    importer = OSMImporter(osm_path)
    data = importer.parse()

    assert "metadata" in data
    assert "intersections" in data
    assert "roads" in data
    assert "buildings" in data

    meta = data["metadata"]
    assert meta["source"] == "OpenStreetMap"
    assert meta["total_roads"] >= 150
    assert meta["total_intersections"] >= 100
    assert meta["total_buildings"] >= 200


def test_osm_roads_have_valid_geometry_and_lanes():
    osm_path = Path(__file__).resolve().parent.parent / "data" / "maps" / "bronx_fordham.osm"
    importer = OSMImporter(osm_path)
    data = importer.parse()

    roads = data["roads"]
    for r in roads[:30]:
        assert "id" in r
        assert "from_node" in r
        assert "to_node" in r
        assert r["lanes"] >= 1
        assert r["length"] > 0.0
        assert len(r["points"]) >= 2
        for pt in r["points"]:
            assert "x" in pt and "y" in pt


def test_osm_buildings_have_polygons_and_heights():
    osm_path = Path(__file__).resolve().parent.parent / "data" / "maps" / "bronx_fordham.osm"
    importer = OSMImporter(osm_path)
    data = importer.parse()

    buildings = data["buildings"]
    for b in buildings[:30]:
        assert "id" in b
        assert "polygon" in b
        assert len(b["polygon"]) >= 3
        assert b["height"] >= 2.0  # valid positive building height in meters
