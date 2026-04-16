"""Tests for GET /api/models"""
import main


def test_models_returns_200(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200


def test_models_response_structure(client):
    data = client.get("/api/models").json()
    assert "models" in data
    assert "current_model" in data
    assert "current_threshold" in data


def test_models_entries_have_required_fields(client):
    models = client.get("/api/models").json()["models"]
    assert len(models) >= 1
    for m in models:
        assert "id" in m
        assert "name" in m
        assert "description" in m
        assert "framework" in m
        assert "is_current" in m


def test_models_exactly_one_is_current(client):
    models = client.get("/api/models").json()["models"]
    current = [m for m in models if m["is_current"]]
    assert len(current) == 1


def test_models_current_model_reflects_global_state(client):
    data = client.get("/api/models").json()
    assert data["current_model"] == main.current_model_type


def test_models_current_threshold_reflects_global_state(client):
    main.current_threshold = 0.75
    data = client.get("/api/models").json()
    assert data["current_threshold"] == 0.75


def test_models_retinanet_is_pytorch(client):
    models = client.get("/api/models").json()["models"]
    retinanet = next((m for m in models if m["id"] == "retinanet"), None)
    assert retinanet is not None
    assert retinanet["framework"] == "pytorch"
