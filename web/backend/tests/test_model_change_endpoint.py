"""Tests for POST /api/model/change"""
import pytest
import main


def _change(client, model_id="retinanet", threshold="0.5"):
    return client.post(
        "/api/model/change",
        data={"model_id": model_id, "threshold": threshold},
    )


def test_change_model_valid_returns_200(client):
    resp = _change(client, "retinanet", "0.5")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_change_model_updates_threshold(client):
    _change(client, "retinanet", "0.8")
    assert main.current_threshold == pytest.approx(0.8)


def test_change_model_updates_model_type(client):
    _change(client, "retinanet", "0.5")
    assert main.current_model_type == "retinanet"


def test_change_model_response_has_required_fields(client):
    data = _change(client, "retinanet", "0.6").json()
    assert "model_id" in data
    assert "model_name" in data
    assert "threshold" in data


def test_change_model_unknown_id_returns_404(client):
    resp = _change(client, "does_not_exist", "0.5")
    assert resp.status_code == 404


def test_change_model_keras_framework_returns_501(client):
    main.AVAILABLE_MODELS["_test_keras"] = {
        "name": "Fake Keras",
        "description": "Test only",
        "type": "keras",
        "weights_path": None,
        "framework": "keras",
    }
    try:
        resp = _change(client, "_test_keras", "0.5")
        assert resp.status_code == 501
    finally:
        main.AVAILABLE_MODELS.pop("_test_keras", None)


def test_change_model_tensorflow_framework_returns_501(client):
    main.AVAILABLE_MODELS["_test_tf"] = {
        "name": "Fake TF",
        "description": "Test only",
        "type": "tensorflow",
        "weights_path": None,
        "framework": "tensorflow",
    }
    try:
        resp = _change(client, "_test_tf", "0.5")
        assert resp.status_code == 501
    finally:
        main.AVAILABLE_MODELS.pop("_test_tf", None)


def test_change_model_non_float_threshold_returns_422(client):
    resp = client.post(
        "/api/model/change",
        data={"model_id": "retinanet", "threshold": "not_a_float"},
    )
    assert resp.status_code == 422
