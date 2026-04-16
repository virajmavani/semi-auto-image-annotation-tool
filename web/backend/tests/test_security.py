"""
Cross-cutting security tests — all path-traversal and boundary checks
consolidated for easy review.
"""
import main


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

def test_upload_dotdot_filename(client, sample_png_bytes):
    """../../etc/passwd — Path().name strips dirs, result is 'passwd' (safe basename)."""
    resp = client.post(
        "/api/upload",
        files={"file": ("../../etc/passwd", sample_png_bytes, "image/png")},
    )
    # Path(filename).name → "passwd"; that basename has no ".." so it passes validate_filename.
    # The file is saved as "passwd" in UPLOAD_DIR (not in /etc). Verify no escape.
    if resp.status_code == 200:
        saved_path = main.UPLOAD_DIR / resp.json()["filename"]
        assert saved_path.is_relative_to(main.UPLOAD_DIR)
        saved_path.unlink(missing_ok=True)
    else:
        assert resp.status_code in (400, 500)


def test_upload_size_limit_enforced(client, mocker):
    """Files over MAX_UPLOAD_SIZE must return 413."""
    mocker.patch.object(main, "MAX_UPLOAD_SIZE", 50)
    data = b"\x00" * 100
    resp = client.post(
        "/api/upload",
        files={"file": ("toobig.png", data, "image/png")},
    )
    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Image serving endpoint
# ---------------------------------------------------------------------------

def test_get_image_encoded_dotdot_returns_400_or_404(client):
    """%2E%2E in path — decoded to '..' by FastAPI, caught by validate_filename."""
    resp = client.get("/api/image/%2E%2Eetc%2Fpasswd")
    assert resp.status_code in (400, 404)


def test_get_image_encoded_slash_returns_400_or_404(client):
    """%2F at start — decoded to '/', caught by validate_filename startswith check."""
    resp = client.get("/api/image/%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)


def test_get_image_backslash_path(client):
    """Backslash path separator attempt."""
    resp = client.get("/api/image/..\\etc\\passwd")
    assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Dataset image endpoint
# ---------------------------------------------------------------------------

def test_dataset_image_encoded_dotdot_returns_400_or_404(client, dataset_dir):
    client.post("/api/dataset/set", data={"directory_path": str(dataset_dir)})
    resp = client.get("/api/dataset/image/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Detect endpoint
# ---------------------------------------------------------------------------

def test_detect_path_outside_allowed(client, mock_read_image):
    """/etc/passwd is outside UPLOAD_DIR and dataset_dir → 400."""
    resp = client.post(
        "/api/detect",
        json={"image_path": "/etc/passwd", "selected_labels": ["person"]},
    )
    assert resp.status_code == 400


def test_detect_path_traversal_via_upload_dir(client, tmp_path, sample_png_bytes, mock_read_image):
    """UPLOAD_DIR/../outside/file resolves outside UPLOAD_DIR → 400."""
    outside = tmp_path / "secret.png"
    outside.write_bytes(sample_png_bytes)
    # Construct a path that looks like it's in UPLOAD_DIR but resolves outside.
    traversal = str(main.UPLOAD_DIR / ".." / outside.name)
    resp = client.post(
        "/api/detect",
        json={"image_path": traversal, "selected_labels": ["person"]},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Browse endpoint
# ---------------------------------------------------------------------------

def test_browse_etc_returns_400(client):
    """/etc is outside home → 400."""
    resp = client.get("/api/browse?path=/etc")
    assert resp.status_code == 400


def test_browse_root_returns_400(client):
    resp = client.get("/api/browse?path=/")
    assert resp.status_code == 400


def test_browse_var_returns_400(client):
    resp = client.get("/api/browse?path=/var")
    assert resp.status_code == 400
