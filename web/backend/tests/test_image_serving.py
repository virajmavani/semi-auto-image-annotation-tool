"""Tests for GET /api/image/{filename}"""


def test_get_existing_image_returns_200(client, uploaded_image):
    resp = client.get(f"/api/image/{uploaded_image}")
    assert resp.status_code == 200


def test_get_image_content_type_is_image(client, uploaded_image):
    resp = client.get(f"/api/image/{uploaded_image}")
    assert resp.headers["content-type"].startswith("image/")


def test_get_nonexistent_image_returns_404(client):
    resp = client.get("/api/image/does_not_exist.png")
    assert resp.status_code == 404


def test_get_image_dotdot_encoded_returns_400(client):
    """URL-encoded '..' in filename is caught by validate_filename."""
    resp = client.get("/api/image/%2E%2Eetc%2Fpasswd")
    assert resp.status_code in (400, 404)


def test_get_image_leading_slash_encoded_returns_400(client):
    """URL-encoded leading '/' is caught by validate_filename."""
    resp = client.get("/api/image/%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)


def test_get_image_backslash_traversal(client):
    """Backslash path traversal attempt."""
    resp = client.get("/api/image/..\\etc\\passwd")
    assert resp.status_code in (400, 404)
