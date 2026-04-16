"""Tests for POST /api/detect"""
import main


def _detect(client, image_path, selected_labels):
    return client.post(
        "/api/detect",
        json={"image_path": image_path, "selected_labels": selected_labels},
    )


def test_detect_filters_by_label(client, uploaded_image, mock_read_image):
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    resp = _detect(client, img_path, ["person"])
    assert resp.status_code == 200
    detections = resp.json()["detections"]
    assert len(detections) == 1
    assert detections[0]["label"] == "person"


def test_detect_empty_selected_labels_returns_none(client, uploaded_image, mock_read_image):
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    resp = _detect(client, img_path, [])
    assert resp.status_code == 200
    assert resp.json()["detections"] == []


def test_detect_all_labels_returns_all_detections(client, uploaded_image, mock_read_image):
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    resp = _detect(client, img_path, ["person", "car"])
    assert resp.status_code == 200
    assert len(resp.json()["detections"]) == 2


def test_detect_detection_fields(client, uploaded_image, mock_read_image):
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    resp = _detect(client, img_path, ["person"])
    det = resp.json()["detections"][0]
    for field in ("x1", "y1", "x2", "y2", "label", "score"):
        assert field in det
    assert isinstance(det["score"], float)


def test_detect_no_model_returns_500(client, uploaded_image, mock_read_image):
    main.current_model = None
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    resp = _detect(client, img_path, ["person"])
    assert resp.status_code == 500


def test_detect_nonexistent_file_returns_404(client, mock_read_image):
    img_path = str(main.UPLOAD_DIR / "does_not_exist.png")
    resp = _detect(client, img_path, ["person"])
    assert resp.status_code == 404


def test_detect_path_outside_allowed_dirs_returns_400(client, tmp_path, sample_png_bytes, mock_read_image):
    """File exists but is outside both UPLOAD_DIR and dataset_dir → 400."""
    outside = tmp_path / "outside.png"
    outside.write_bytes(sample_png_bytes)
    resp = _detect(client, str(outside), ["person"])
    assert resp.status_code == 400


def test_detect_path_in_dataset_dir_returns_200(client, dataset_dir, mock_read_image):
    main.current_dataset_dir = dataset_dir
    img_path = str(dataset_dir / "img1.png")
    resp = _detect(client, img_path, ["person"])
    assert resp.status_code == 200


def test_detect_calls_preprocess_and_predict(client, uploaded_image, mock_model, mock_read_image):
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    _detect(client, img_path, ["person"])
    mock_model.preprocess_image.assert_called_once()
    mock_model.predict.assert_called_once()


def test_detect_zero_shot_passes_text_queries(client, uploaded_image, mock_model, mock_read_image):
    """Zero-shot model: selected_labels forwarded as text_queries, no post-filtering."""
    mock_model.is_zero_shot.return_value = True
    mock_model.predict.return_value = [
        {"box": [10, 20, 100, 200], "label": "vintage teapot", "score": 0.85}
    ]
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    resp = _detect(client, img_path, ["vintage teapot"])
    assert resp.status_code == 200
    detections = resp.json()["detections"]
    assert len(detections) == 1
    assert detections[0]["label"] == "vintage teapot"
    _, kwargs = mock_model.predict.call_args
    assert kwargs.get("text_queries") == ["vintage teapot"]


def test_detect_zero_shot_empty_labels_returns_empty(client, uploaded_image, mock_model, mock_read_image):
    """Zero-shot model with no queries returns empty detections immediately."""
    mock_model.is_zero_shot.return_value = True
    img_path = str(main.UPLOAD_DIR / uploaded_image)
    resp = _detect(client, img_path, [])
    assert resp.status_code == 200
    assert resp.json()["detections"] == []
    mock_model.predict.assert_not_called()
