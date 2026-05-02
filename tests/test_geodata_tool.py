"""Tests for app.tools.geodata_tool — AOI resolution, provider clients,
and the parallel orchestrator. All HTTP is mocked; no live API calls.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ── AOI resolution ──────────────────────────────────────────────────


class TestResolveCountry:

    def test_iso3_hits_builtin_table(self):
        from app.tools.geodata_tool import _resolve_country
        out = _resolve_country("FIN")
        assert out is not None
        assert out["iso3"] == "FIN"
        assert out["name"] == "Finland"
        assert len(out["bbox"]) == 4

    def test_iso3_lowercase_works(self):
        from app.tools.geodata_tool import _resolve_country
        out = _resolve_country("fin")
        assert out is not None and out["iso3"] == "FIN"

    def test_english_name_hits_builtin_table(self):
        from app.tools.geodata_tool import _resolve_country
        out = _resolve_country("Brazil")
        assert out is not None and out["iso3"] == "BRA"

    def test_alias_resolves(self):
        from app.tools.geodata_tool import _resolve_country
        for alias in ("US", "uk", "DRC"):
            out = _resolve_country(alias)
            assert out is not None, f"alias {alias!r} did not resolve"

    def test_unknown_country_falls_back_to_nominatim(self, monkeypatch):
        # Bypass the lru cache by clearing it.
        from app.tools import geodata_tool as gt
        gt._NOMINATIM_CACHE.clear()

        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = [
            {"display_name": "Andorra", "boundingbox": ["42.4", "42.7", "1.4", "1.8"]}
        ]
        with patch.object(gt._session, "get", return_value=fake_resp) as m:
            out = gt._resolve_country("Andorra")
        assert out is not None
        assert out["bbox"] == [1.4, 42.4, 1.8, 42.7]  # [W,S,E,N]
        m.assert_called_once()

    def test_nominatim_failure_returns_none(self, monkeypatch):
        from app.tools import geodata_tool as gt
        gt._NOMINATIM_CACHE.clear()
        with patch.object(gt._session, "get", side_effect=RuntimeError("boom")):
            assert gt._resolve_country("XYZ-not-a-country") is None


class TestResolveAOI:

    def test_country_input_produces_polygon(self):
        from app.tools.geodata_tool import _resolve_aoi
        aoi = _resolve_aoi(country="EST", bbox=None, geojson=None)
        assert aoi["name"] == "Estonia"
        assert aoi["geometry"]["type"] == "Polygon"
        assert len(aoi["bbox"]) == 4

    def test_bbox_input_synthesises_polygon(self):
        from app.tools.geodata_tool import _resolve_aoi
        aoi = _resolve_aoi(country=None, bbox=[10.0, 50.0, 11.0, 51.0], geojson=None)
        assert aoi["bbox"] == [10.0, 50.0, 11.0, 51.0]
        coords = aoi["geometry"]["coordinates"][0]
        # Closed ring of 5 corners.
        assert len(coords) == 5
        assert coords[0] == coords[-1]

    def test_geojson_input_computes_bbox(self):
        from app.tools.geodata_tool import _resolve_aoi
        geom = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 0], [2, 3], [0, 3], [0, 0]]],
        }
        aoi = _resolve_aoi(country=None, bbox=None, geojson=geom)
        assert aoi["bbox"] == [0, 0, 2, 3]

    def test_no_input_raises(self):
        from app.tools.geodata_tool import _resolve_aoi
        with pytest.raises(ValueError):
            _resolve_aoi(country=None, bbox=None, geojson=None)

    def test_bad_bbox_length_raises(self):
        from app.tools.geodata_tool import _resolve_aoi
        with pytest.raises(ValueError):
            _resolve_aoi(country=None, bbox=[1, 2, 3], geojson=None)

    def test_unknown_country_raises(self, monkeypatch):
        from app.tools import geodata_tool as gt
        gt._NOMINATIM_CACHE.clear()
        with patch.object(gt._session, "get", side_effect=RuntimeError("net")):
            with pytest.raises(ValueError):
                gt._resolve_aoi(country="Atlantis", bbox=None, geojson=None)


# ── Global Forest Watch ────────────────────────────────────────────


class TestGFW:

    def test_list_datasets_ok(self):
        from app.tools import geodata_tool as gt
        fake = MagicMock()
        fake.raise_for_status = MagicMock()
        fake.json.return_value = {"data": [
            {"dataset": "umd_tree_cover_loss", "metadata": {"title": "TCL"}},
            {"dataset": "gfw_integrated_alerts", "metadata": {"title": "Alerts"}},
        ]}
        with patch.object(gt._session, "get", return_value=fake):
            out = gt.gfw_list_datasets()
        assert out["ok"] is True
        assert out["count"] == 2
        assert out["datasets"][0]["id"] == "umd_tree_cover_loss"

    def test_list_datasets_failure_isolated(self):
        from app.tools import geodata_tool as gt
        with patch.object(gt._session, "get", side_effect=RuntimeError("503")):
            out = gt.gfw_list_datasets()
        assert out["ok"] is False
        assert "RuntimeError" in out["error"]

    def test_query_aoi_handles_geostore_failure(self):
        from app.tools import geodata_tool as gt
        with patch.object(gt._session, "post", side_effect=RuntimeError("auth")):
            out = gt.gfw_query_aoi({"geometry": {"type": "Polygon", "coordinates": []}})
        assert out["ok"] is False


# ── Copernicus Data Space ──────────────────────────────────────────


class TestCDSE:

    def test_list_collections_ok(self):
        from app.tools import geodata_tool as gt
        fake = MagicMock()
        fake.raise_for_status = MagicMock()
        fake.json.return_value = {"collections": [
            {"id": "SENTINEL-2", "title": "Sentinel-2", "description": "Optical L2A"},
            {"id": "SENTINEL-1", "title": "Sentinel-1", "description": "SAR GRD"},
        ]}
        with patch.object(gt._session, "get", return_value=fake):
            out = gt.cdse_list_collections()
        assert out["ok"] is True
        assert out["count"] == 2
        ids = {c["id"] for c in out["collections"]}
        assert "SENTINEL-2" in ids and "SENTINEL-1" in ids

    def test_search_aoi_passes_bbox_and_dates(self):
        from app.tools import geodata_tool as gt
        fake = MagicMock()
        fake.raise_for_status = MagicMock()
        fake.json.return_value = {
            "context": {"matched": 42},
            "features": [{
                "id": "S2A_MSIL2A_xxx",
                "collection": "SENTINEL-2",
                "properties": {"datetime": "2024-06-01T10:00Z", "eo:cloud_cover": 12.5},
                "links": [{"rel": "self", "href": "https://x/y"}],
            }],
        }
        aoi = {"bbox": [10, 50, 11, 51], "geometry": {}, "name": "test"}
        with patch.object(gt._session, "post", return_value=fake) as m:
            out = gt.cdse_search_aoi(aoi, date_from="2024-06-01", date_to="2024-06-30")
        assert out["ok"] is True
        assert out["matched"] == 42
        # Confirm body shape sent to STAC.
        sent_body = m.call_args.kwargs["json"]
        assert sent_body["bbox"] == [10, 50, 11, 51]
        assert sent_body["datetime"] == "2024-06-01/2024-06-30"
        assert sent_body["collections"] == ["SENTINEL-2"]


# ── GEE provider (no curated discovery — delegated to dataset_search) ─


class TestGEE:

    def test_query_aoi_without_ee_init_returns_error(self, monkeypatch):
        from app.tools import geodata_tool as gt
        with patch("app.tools.gee_tool._ensure_initialised", return_value=(False, "no creds")):
            out = gt.gee_query_aoi({"bbox": [10, 50, 11, 51], "geometry": {}, "name": "x"})
        assert out["ok"] is False
        assert "no creds" in out["error"]

    def test_discover_does_not_enumerate_gee(self, monkeypatch):
        # GEE catalog discovery is intentionally delegated to
        # dataset_search; default discover_all should not call any GEE
        # listing function.
        from app.tools import geodata_tool as gt
        with patch.object(gt, "gfw_list_datasets", return_value={"ok": True, "count": 0}), \
             patch.object(gt, "cdse_list_collections", return_value={"ok": True, "count": 0}):
            out = gt.discover_all()
        assert "gee" not in out
        assert set(out.keys()) == {"gfw", "cdse"}

    def test_explicit_gee_request_returns_pointer(self, monkeypatch):
        # If a caller insists on providers=['gee'], we surface a
        # pointer to dataset_search rather than dropping silently.
        from app.tools import geodata_tool as gt
        out = gt.discover_all(providers=["gee"])
        assert "gee" in out
        assert out["gee"]["ok"] is False
        assert "dataset_search" in out["gee"]["info"]


# ── Parallel orchestrator ──────────────────────────────────────────


class TestFetchAll:

    def test_fans_out_to_all_three_providers(self, monkeypatch):
        from app.tools import geodata_tool as gt
        monkeypatch.setattr(gt, "gee_query_aoi", lambda *a, **k: {"ok": True, "src": "gee"})
        monkeypatch.setattr(gt, "gfw_query_aoi", lambda *a, **k: {"ok": True, "src": "gfw"})
        monkeypatch.setattr(gt, "cdse_search_aoi", lambda *a, **k: {"ok": True, "src": "cdse"})

        out = gt.fetch_all(country="EST", date_from="2024-01-01", date_to="2024-12-31")
        assert set(out["providers"].keys()) == {"gee", "gfw", "cdse"}
        for p in ("gee", "gfw", "cdse"):
            assert out["providers"][p]["ok"] is True

    def test_one_provider_failure_does_not_block_others(self, monkeypatch):
        from app.tools import geodata_tool as gt
        def boom(*a, **k):
            raise RuntimeError("dead")
        monkeypatch.setattr(gt, "gee_query_aoi", boom)
        monkeypatch.setattr(gt, "gfw_query_aoi", lambda *a, **k: {"ok": True})
        monkeypatch.setattr(gt, "cdse_search_aoi", lambda *a, **k: {"ok": True})

        out = gt.fetch_all(country="FIN")
        assert out["providers"]["gee"]["ok"] is False
        assert "RuntimeError" in out["providers"]["gee"]["error"]
        assert out["providers"]["gfw"]["ok"] is True
        assert out["providers"]["cdse"]["ok"] is True

    def test_providers_filter_respected(self, monkeypatch):
        from app.tools import geodata_tool as gt
        called: dict[str, int] = {"gee": 0, "gfw": 0, "cdse": 0}
        monkeypatch.setattr(gt, "gee_query_aoi",
                            lambda *a, **k: (called.__setitem__("gee", called["gee"] + 1) or {"ok": True}))
        monkeypatch.setattr(gt, "gfw_query_aoi",
                            lambda *a, **k: (called.__setitem__("gfw", called["gfw"] + 1) or {"ok": True}))
        monkeypatch.setattr(gt, "cdse_search_aoi",
                            lambda *a, **k: (called.__setitem__("cdse", called["cdse"] + 1) or {"ok": True}))

        out = gt.fetch_all(bbox=[10, 50, 11, 51], providers=["cdse", "gfw"])
        assert called == {"gee": 0, "gfw": 1, "cdse": 1}
        assert "gee" not in out["providers"]


# ── CrewAI factory ─────────────────────────────────────────────────


class TestFactory:

    def test_factory_returns_two_tools(self):
        from app.tools.geodata_tool import create_geodata_tools
        tools = create_geodata_tools()
        # If crewai/pydantic are missing in this env, the factory returns
        # []; otherwise we expect the discover + fetch pair.
        if not tools:
            pytest.skip("crewai/pydantic not importable in this env")
        names = {t.name for t in tools}
        assert names == {"geodata_discover", "geodata_fetch"}

    def test_fetch_tool_rejects_missing_aoi(self):
        from app.tools.geodata_tool import create_geodata_tools
        tools = create_geodata_tools()
        if not tools:
            pytest.skip("crewai/pydantic not importable in this env")
        fetch = next(t for t in tools if t.name == "geodata_fetch")
        out = fetch._run()  # no country/bbox/geojson
        assert "bad AOI input" in out
