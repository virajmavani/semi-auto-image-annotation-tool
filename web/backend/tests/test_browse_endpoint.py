"""Tests for GET /api/browse"""
from pathlib import Path


def test_browse_home_returns_200(client):
    resp = client.get("/api/browse?path=~")
    assert resp.status_code == 200


def test_browse_home_response_structure(client):
    data = client.get("/api/browse?path=~").json()
    assert "current_path" in data
    assert "entries" in data
    assert "parent_path" in data


def test_browse_home_current_path_is_home(client):
    data = client.get("/api/browse?path=~").json()
    assert data["current_path"] == str(Path.home())


def test_browse_home_entries_have_name_and_path(client):
    data = client.get("/api/browse?path=~").json()
    for entry in data["entries"]:
        assert "name" in entry
        assert "path" in entry


def test_browse_home_no_hidden_dirs_in_entries(client):
    """Entries starting with '.' must be excluded."""
    data = client.get("/api/browse?path=~").json()
    for entry in data["entries"]:
        assert not entry["name"].startswith(".")


def test_browse_home_with_explicit_path(client):
    home = str(Path.home())
    resp = client.get(f"/api/browse?path={home}")
    assert resp.status_code == 200


def test_browse_outside_home_returns_400(client):
    resp = client.get("/api/browse?path=/etc")
    assert resp.status_code == 400


def test_browse_root_returns_400(client):
    resp = client.get("/api/browse?path=/")
    assert resp.status_code == 400


def test_browse_nonexistent_path_returns_400(client):
    resp = client.get("/api/browse?path=/nonexistent/path/xyz123")
    assert resp.status_code == 400
