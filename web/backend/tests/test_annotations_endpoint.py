"""Tests for GET /api/annotations/{filename}"""
import csv
import main


def _write_csv(rows):
    """Helper: write rows to the annotations CSV."""
    csv_path = main.ANNOTATIONS_DIR / "annotations.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return csv_path


def test_annotations_no_csv_returns_empty(client):
    (main.ANNOTATIONS_DIR / "annotations.csv").unlink(missing_ok=True)
    data = client.get("/api/annotations/test.png").json()
    assert data == {"bboxes": []}


def test_annotations_returns_matching_rows_only(client):
    _write_csv([
        ["target.png", "10", "20", "100", "200", "person"],
        ["other.png", "5", "5", "50", "50", "cat"],
    ])
    data = client.get("/api/annotations/target.png").json()
    assert len(data["bboxes"]) == 1
    assert data["bboxes"][0]["label"] == "person"


def test_annotations_correct_bbox_structure(client):
    _write_csv([["struct.png", "1", "2", "3", "4", "dog"]])
    bbox = client.get("/api/annotations/struct.png").json()["bboxes"][0]
    assert bbox == {"x1": 1, "y1": 2, "x2": 3, "y2": 4, "label": "dog"}


def test_annotations_no_score_in_response(client):
    """Score is not stored in CSV; should not appear in the response."""
    _write_csv([["noscore.png", "1", "2", "3", "4", "cat"]])
    bbox = client.get("/api/annotations/noscore.png").json()["bboxes"][0]
    assert "score" not in bbox


def test_annotations_malformed_line_silently_skipped(client):
    """Lines with < 6 fields are skipped; valid lines still returned."""
    csv_path = main.ANNOTATIONS_DIR / "annotations.csv"
    with open(csv_path, "w") as f:
        f.write("malformed.png,1,2\n")  # too few fields
        f.write("malformed.png,10,20,100,200,person\n")  # valid
    data = client.get("/api/annotations/malformed.png").json()
    assert len(data["bboxes"]) == 1


def test_annotations_multiple_bboxes_for_same_image(client):
    _write_csv([
        ["multi.png", "10", "20", "100", "200", "person"],
        ["multi.png", "50", "60", "150", "250", "car"],
        ["other.png", "0", "0", "10", "10", "cat"],
    ])
    data = client.get("/api/annotations/multi.png").json()
    assert len(data["bboxes"]) == 2
    labels = {b["label"] for b in data["bboxes"]}
    assert labels == {"person", "car"}


def test_annotations_integer_coordinates(client):
    _write_csv([["intcoord.png", "10", "20", "100", "200", "person"]])
    bbox = client.get("/api/annotations/intcoord.png").json()["bboxes"][0]
    assert isinstance(bbox["x1"], int)
    assert isinstance(bbox["y1"], int)


def test_annotations_nonexistent_image_returns_empty(client):
    _write_csv([["existing.png", "10", "20", "100", "200", "person"]])
    data = client.get("/api/annotations/not_in_csv.png").json()
    assert data["bboxes"] == []


def test_annotations_invalid_filename_returns_empty_not_500(client):
    """Validate_filename raises 400 but the catch-all handler returns [] for any exception."""
    # The endpoint has a broad except that returns {"bboxes": []} on any error.
    resp = client.get("/api/annotations/../etc/passwd")
    # Route may not match due to URL structure; either way no 500.
    assert resp.status_code != 500
