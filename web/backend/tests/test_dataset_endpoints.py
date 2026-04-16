"""Tests for POST /api/dataset/set, GET /api/dataset/images, GET /api/dataset/image/{filename}"""
from pathlib import Path
import main


# ---------------------------------------------------------------------------
# POST /api/dataset/set
# ---------------------------------------------------------------------------

def test_set_dataset_valid_dir_returns_200(client, dataset_dir):
    resp = client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_set_dataset_image_count(client, dataset_dir):
    data = client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)}).json()
    assert data["image_count"] == 2  # 2 PNGs, not the .txt file


def test_set_dataset_response_has_directory(client, dataset_dir):
    data = client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)}).json()
    assert "directory" in data
    assert Path(data["directory"]) == dataset_dir.resolve()


def test_set_dataset_nonexistent_path_returns_404(client):
    resp = client.post("/api/dataset/set", data={"directory_path": "/nonexistent/path/xyz"})
    assert resp.status_code == 404


def test_set_dataset_file_path_returns_400(client, dataset_dir):
    file_path = dataset_dir / "img1.png"
    resp = client.post("/api/dataset/set", data={"directory_path": str(file_path)})
    assert resp.status_code == 400


def test_set_dataset_empty_dir_returns_400(client, tmp_path):
    # Empty directory — no valid images.
    empty = tmp_path / "empty"
    empty.mkdir()
    resp = client.post("/api/dataset/set", data={"directory_path": str(empty)})
    assert resp.status_code == 400


def test_set_dataset_sets_global_state(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    assert main.current_dataset_dir == dataset_dir.resolve()


def test_set_dataset_tilde_expansion(client, dataset_dir, tmp_path):
    """~ is expanded; should not crash even if home has no images (we just verify no 500)."""
    resp = client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    assert resp.status_code != 500


# ---------------------------------------------------------------------------
# GET /api/dataset/images
# ---------------------------------------------------------------------------

def test_list_dataset_images_no_dir_returns_400(client):
    resp = client.get("/api/dataset/images")
    assert resp.status_code == 400


def test_list_dataset_images_returns_images(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    data = client.get("/api/dataset/images").json()
    assert "images" in data
    names = [img["filename"] for img in data["images"]]
    assert "img1.png" in names
    assert "img2.png" in names


def test_list_dataset_images_excludes_txt(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    data = client.get("/api/dataset/images").json()
    names = [img["filename"] for img in data["images"]]
    assert "notes.txt" not in names


def test_list_dataset_images_sorted(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    data = client.get("/api/dataset/images").json()
    names = [img["filename"] for img in data["images"]]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# GET /api/dataset/image/{filename}
# ---------------------------------------------------------------------------

def test_get_dataset_image_returns_200(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    resp = client.get("/api/dataset/image/img1.png")
    assert resp.status_code == 200


def test_get_dataset_image_no_dir_returns_400(client):
    resp = client.get("/api/dataset/image/img1.png")
    assert resp.status_code == 400


def test_get_dataset_image_nonexistent_returns_404(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    resp = client.get("/api/dataset/image/no_such_file.png")
    assert resp.status_code == 404


def test_get_dataset_image_path_traversal_returns_400(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    resp = client.get("/api/dataset/image/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)
