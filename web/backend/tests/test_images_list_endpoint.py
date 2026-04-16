"""Tests for GET /api/images"""
import main


def test_list_images_returns_200(client):
    resp = client.get("/api/images")
    assert resp.status_code == 200


def test_list_images_empty_dir(client):
    # Clear any leftover files from other tests.
    for f in main.UPLOAD_DIR.glob("*"):
        f.unlink(missing_ok=True)
    data = client.get("/api/images").json()
    assert data["images"] == []


def test_list_images_after_upload(client, uploaded_image):
    data = client.get("/api/images").json()
    names = [img["filename"] for img in data["images"]]
    assert uploaded_image in names


def test_list_images_entry_has_required_fields(client, uploaded_image):
    data = client.get("/api/images").json()
    entry = next(img for img in data["images"] if img["filename"] == uploaded_image)
    assert "filename" in entry
    assert "path" in entry
    assert "width" in entry
    assert "height" in entry
    assert isinstance(entry["width"], int)
    assert isinstance(entry["height"], int)


def test_list_images_excludes_txt_files(client):
    # Drop a .txt file into UPLOAD_DIR; it must not appear in the listing.
    txt = main.UPLOAD_DIR / "readme.txt"
    txt.write_text("hello")
    try:
        data = client.get("/api/images").json()
        names = [img["filename"] for img in data["images"]]
        assert "readme.txt" not in names
    finally:
        txt.unlink(missing_ok=True)


def test_list_images_dimensions_correct(client, uploaded_image):
    data = client.get("/api/images").json()
    entry = next(img for img in data["images"] if img["filename"] == uploaded_image)
    # sample_png_bytes fixture creates a 100x80 image.
    assert entry["width"] == 100
    assert entry["height"] == 80
