"""Tests for GET /api/labels"""
import main


def test_labels_returns_200(client):
    resp = client.get("/api/labels")
    assert resp.status_code == 200


def test_labels_returns_fake_labels(client, mock_model):
    data = client.get("/api/labels").json()
    assert data["labels"] == mock_model.get_labels.return_value


def test_labels_calls_get_labels_once(client, mock_model):
    client.get("/api/labels")
    mock_model.get_labels.assert_called_once()


def test_labels_no_model_returns_500(client):
    main.current_model = None
    resp = client.get("/api/labels")
    assert resp.status_code == 500


def test_labels_contains_background(client):
    data = client.get("/api/labels").json()
    assert "__background__" in data["labels"]


def test_labels_contains_common_classes(client):
    data = client.get("/api/labels").json()
    labels = data["labels"]
    assert "person" in labels
    assert "car" in labels
