"""Tests for POST /api/save"""
import csv
import json
import xml.etree.ElementTree as ET
import main


def _save(client, image_name="test.png", bboxes=None, width=640, height=480):
    if bboxes is None:
        bboxes = [
            {"x1": 10, "y1": 20, "x2": 100, "y2": 200, "label": "person", "score": 0.9},
        ]
    return client.post(
        "/api/save",
        json={
            "image_name": image_name,
            "bboxes": bboxes,
            "image_width": width,
            "image_height": height,
        },
    )


def test_save_returns_200(client):
    resp = _save(client)
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_save_creates_csv(client):
    (main.ANNOTATIONS_DIR / "annotations.csv").unlink(missing_ok=True)
    _save(client, image_name="csv_test.png")
    csv_path = main.ANNOTATIONS_DIR / "annotations.csv"
    assert csv_path.exists()


def test_save_csv_row_format(client):
    """CSV row: image_name,x1,y1,x2,y2,label (score is NOT written)."""
    csv_path = main.ANNOTATIONS_DIR / "annotations.csv"
    csv_path.unlink(missing_ok=True)

    _save(
        client,
        image_name="rowformat.png",
        bboxes=[{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "label": "cat", "score": 0.8}],
    )

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    row = next(r for r in rows if r[0] == "rowformat.png")
    assert row == ["rowformat.png", "1", "2", "3", "4", "cat"]


def test_save_creates_voc_xml(client):
    _save(client, image_name="voc_test.png")
    xml_path = main.ANNOTATIONS_DIR / "annotations_voc" / "voc_test.xml"
    assert xml_path.exists()


def test_save_xml_structure(client):
    _save(
        client,
        image_name="xmlstruct.png",
        bboxes=[{"x1": 5, "y1": 10, "x2": 50, "y2": 100, "label": "dog", "score": 0.7}],
        width=320,
        height=240,
    )
    xml_path = main.ANNOTATIONS_DIR / "annotations_voc" / "xmlstruct.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.tag == "annotation"
    obj = root.find("object")
    assert obj is not None
    assert obj.find("name").text == "dog"
    bndbox = obj.find("bndbox")
    assert bndbox.find("xmin").text == "5"
    assert bndbox.find("ymin").text == "10"
    assert bndbox.find("xmax").text == "50"
    assert bndbox.find("ymax").text == "100"


def test_save_csv_appends_on_second_call(client):
    csv_path = main.ANNOTATIONS_DIR / "annotations.csv"
    csv_path.unlink(missing_ok=True)

    _save(client, image_name="append1.png")
    _save(client, image_name="append2.png")

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    names = [r[0] for r in rows if r]
    assert "append1.png" in names
    assert "append2.png" in names


def test_save_empty_bboxes_produces_xml_no_objects(client):
    _save(client, image_name="empty_bbox.png", bboxes=[])
    xml_path = main.ANNOTATIONS_DIR / "annotations_voc" / "empty_bbox.xml"
    assert xml_path.exists()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.find("object") is None


def test_save_image_name_with_path_prefix_uses_stem_for_xml(client):
    """XML filename should use Path.stem of image_name."""
    _save(client, image_name="subdir/stemtest.png")
    xml_path = main.ANNOTATIONS_DIR / "annotations_voc" / "stemtest.xml"
    assert xml_path.exists()


def test_save_creates_jsonl(client):
    jsonl_path = main.ANNOTATIONS_DIR / "annotations.jsonl"
    jsonl_path.unlink(missing_ok=True)
    _save(client, image_name="jsonl_test.png")
    assert jsonl_path.exists()


def test_save_jsonl_record_format(client):
    jsonl_path = main.ANNOTATIONS_DIR / "annotations.jsonl"
    jsonl_path.unlink(missing_ok=True)

    _save(
        client,
        image_name="jsonl_format.png",
        bboxes=[{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "label": "cat", "score": 0.8}],
        width=640,
        height=480,
    )

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["image"] == "jsonl_format.png"
    assert record["width"] == 640
    assert record["height"] == 480
    assert len(record["annotations"]) == 1
    ann = record["annotations"][0]
    assert ann["label"] == "cat"
    assert ann["bbox"] == [1, 2, 3, 4]
    assert ann["score"] == 0.8


def test_save_jsonl_omits_score_when_none(client):
    jsonl_path = main.ANNOTATIONS_DIR / "annotations.jsonl"
    jsonl_path.unlink(missing_ok=True)

    _save(
        client,
        image_name="jsonl_noscore.png",
        bboxes=[{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "label": "dog", "score": None}],
    )

    record = json.loads(jsonl_path.read_text().strip())
    assert "score" not in record["annotations"][0]


def test_save_jsonl_appends_on_second_call(client):
    jsonl_path = main.ANNOTATIONS_DIR / "annotations.jsonl"
    jsonl_path.unlink(missing_ok=True)

    _save(client, image_name="jsonl_append1.png")
    _save(client, image_name="jsonl_append2.png")

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
    images = [json.loads(l)["image"] for l in lines]
    assert "jsonl_append1.png" in images
    assert "jsonl_append2.png" in images


def test_save_response_includes_jsonl_path(client):
    resp = _save(client)
    assert resp.status_code == 200
    assert "jsonl_path" in resp.json()
