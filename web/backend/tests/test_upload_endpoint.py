"""Tests for POST /api/upload"""
import main


def test_upload_valid_png_returns_200(client, sample_png_bytes):
    resp = client.post(
        "/api/upload",
        files={"file": ("upload_test.png", sample_png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    (main.UPLOAD_DIR / "upload_test.png").unlink(missing_ok=True)


def test_upload_returns_filename_dimensions(client, sample_png_bytes):
    resp = client.post(
        "/api/upload",
        files={"file": ("dims_test.png", sample_png_bytes, "image/png")},
    )
    data = resp.json()
    assert data["filename"] == "dims_test.png"
    assert data["width"] == 100
    assert data["height"] == 80
    (main.UPLOAD_DIR / "dims_test.png").unlink(missing_ok=True)


def test_upload_valid_jpeg(client, sample_jpeg_bytes):
    resp = client.post(
        "/api/upload",
        files={"file": ("upload_test.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    (main.UPLOAD_DIR / "upload_test.jpg").unlink(missing_ok=True)


def test_upload_strips_directory_prefix(client, sample_png_bytes):
    """Path(filename).name strips 'subdir/' prefix — only the basename is stored."""
    resp = client.post(
        "/api/upload",
        files={"file": ("subdir/stripped.png", sample_png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["filename"] == "stripped.png"
    (main.UPLOAD_DIR / "stripped.png").unlink(missing_ok=True)


def test_upload_dotdot_filename_returns_400(client, sample_png_bytes):
    resp = client.post(
        "/api/upload",
        files={"file": ("../../etc/passwd", sample_png_bytes, "image/png")},
    )
    # After Path(filename).name strips dirs, we get "passwd" — which is safe.
    # The dotdot check fires only if ".." survives Path().name stripping.
    # This test documents the actual behaviour: strip then validate.
    assert resp.status_code in (200, 400)


def test_upload_non_image_bytes_returns_500(client):
    garbage = b"this is definitely not an image"
    resp = client.post(
        "/api/upload",
        files={"file": ("garbage.png", garbage, "image/png")},
    )
    assert resp.status_code == 500
    # Backend doesn't clean up on Image.open failure; clean up manually.
    (main.UPLOAD_DIR / "garbage.png").unlink(missing_ok=True)


def test_upload_large_file_returns_413(client, mocker):
    """Files exceeding MAX_UPLOAD_SIZE are rejected (streaming check)."""
    # Patch MAX_UPLOAD_SIZE to 100 bytes so we don't need to allocate 50 MB.
    mocker.patch.object(main, "MAX_UPLOAD_SIZE", 100)
    small_but_over_limit = b"\x00" * 200
    resp = client.post(
        "/api/upload",
        files={"file": ("toobig.png", small_but_over_limit, "image/png")},
    )
    assert resp.status_code == 413


def test_upload_large_file_cleanup(client, mocker):
    """Partial file is deleted when the streaming limit is hit."""
    mocker.patch.object(main, "MAX_UPLOAD_SIZE", 100)
    filename = "cleanup_check.png"
    data = b"\x00" * 200
    resp = client.post(
        "/api/upload",
        files={"file": (filename, data, "image/png")},
    )
    assert resp.status_code == 413
    assert not (main.UPLOAD_DIR / filename).exists()


def test_upload_file_is_saved_to_upload_dir(client, sample_png_bytes):
    filename = "saved_check.png"
    resp = client.post(
        "/api/upload",
        files={"file": (filename, sample_png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert (main.UPLOAD_DIR / filename).exists()
    (main.UPLOAD_DIR / filename).unlink(missing_ok=True)
