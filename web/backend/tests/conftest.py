"""
Shared fixtures for the Anno-Mage web backend test suite.

Bootstrap order (all at module level, before any fixture):
  1. Fake torch / torchvision added to sys.modules so the models package
     can be imported without GPU libraries installed.
  2. web/backend/main.py loaded by file path (not name-lookup) so it is
     never confused with the repo-root Tkinter desktop main.py.
  3. sys.modules["main"] set to the backend module so that subsequent
     `import main` statements in test files return the right object.

Fixture design:
  - mock_model  (session): MagicMock implementing AbstractModel's interface.
  - client      (session): TestClient with ModelFactory patched and I/O
                           redirected to tmp directories.
  - reset_global_state (function, autouse): resets mutable globals and mock
                           call history between every test.
  - uploaded_image (function): uploads a PNG via the API, yields filename,
                           deletes the file on teardown.
  - dataset_dir (function): temp dir with 2 PNGs + 1 .txt.
  - mock_read_image (function): patches torchvision.io.image.read_image.
"""

# ---------------------------------------------------------------------------
# Step 1 — Fake GPU libraries (torch / torchvision) before any model import
# ---------------------------------------------------------------------------
import importlib.util
import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _fake_module(name: str, **attrs):
    """Create a fake module, register it in sys.modules, and return it."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


# Register fakes only if the real libraries aren't available.
if "torch" not in sys.modules:
    _fake_module("torch")

for _name in ("torchvision", "torchvision.io", "torchvision.transforms",
              "torchvision.transforms.functional"):
    _fake_module(_name)

_fake_module(
    "torchvision.io.image",
    read_image=MagicMock(return_value=MagicMock()),
)
_fake_module(
    "torchvision.models",
)
_fake_module(
    "torchvision.models.detection",
    retinanet_resnet50_fpn_v2=MagicMock(),
    RetinaNet_ResNet50_FPN_V2_Weights=MagicMock(),
)

# ---------------------------------------------------------------------------
# Step 2 — Load web/backend/main.py by absolute path, register as "main"
#           so test-file `import main` returns the FastAPI backend, not the
#           Tkinter desktop app at the repo root.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent.parent      # …/web/backend
_REPO_DIR = _BACKEND_DIR.parent.parent           # repo root
_BACKEND_MAIN = _BACKEND_DIR / "main.py"

if "main" not in sys.modules:
    # Ensure models/ (at repo root) is importable before exec'ing main.py
    if str(_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(_REPO_DIR))

    _spec = importlib.util.spec_from_file_location("main", str(_BACKEND_MAIN))
    _main_mod = importlib.util.module_from_spec(_spec)
    sys.modules["main"] = _main_mod           # register before exec to handle any self-refs
    _spec.loader.exec_module(_main_mod)        # runs module-level code (routes, globals…)

from fastapi.testclient import TestClient     # noqa: E402 (after sys.modules setup)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
FAKE_LABELS = ["__background__", "person", "bicycle", "car", "cat", "dog"]

FAKE_DETECTIONS = [
    {"box": [10, 20, 100, 200], "label": "person", "score": 0.92},
    {"box": [50, 60, 150, 250], "label": "car",    "score": 0.75},
]


# ---------------------------------------------------------------------------
# Session-scoped mock model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_model():
    model = MagicMock()
    model.get_labels.return_value = FAKE_LABELS
    model.preprocess_image.return_value = MagicMock()
    model.predict.return_value = list(FAKE_DETECTIONS)
    return model


# ---------------------------------------------------------------------------
# In-memory image bytes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_png_bytes():
    """100×80 white PNG as raw bytes."""
    img = Image.new("RGB", (100, 80), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_jpeg_bytes():
    """100×80 white JPEG as raw bytes."""
    img = Image.new("RGB", (100, 80), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session-scoped TestClient — ModelFactory patched, I/O redirected
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client(mock_model, tmp_path_factory):
    import main  # noqa: PLC0415 — already in sys.modules; fast lookup

    session_tmp = tmp_path_factory.mktemp("annomage")
    uploads = session_tmp / "uploads"
    annotations = session_tmp / "annotations"
    uploads.mkdir()
    annotations.mkdir()

    main.UPLOAD_DIR = uploads
    main.ANNOTATIONS_DIR = annotations

    with patch("main.ModelFactory.create_model", return_value=mock_model):
        with TestClient(main.app, raise_server_exceptions=True) as c:
            main.current_model = mock_model   # override whatever startup set
            yield c


# ---------------------------------------------------------------------------
# Function-scoped autouse: reset mutable globals between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_global_state(client, mock_model):  # noqa: ARG001
    import main  # noqa: PLC0415

    main.current_model = mock_model
    main.current_model_type = "retinanet"
    main.current_threshold = 0.5
    main.current_dataset_dir = None

    mock_model.reset_mock()
    mock_model.get_labels.return_value = FAKE_LABELS
    mock_model.preprocess_image.return_value = MagicMock()
    mock_model.predict.return_value = list(FAKE_DETECTIONS)

    yield


# ---------------------------------------------------------------------------
# Upload helper fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def uploaded_image(client, sample_png_bytes):
    """POST a PNG via /api/upload; yield filename; delete the file on teardown."""
    resp = client.post(
        "/api/upload",
        files={"file": ("test_image.png", sample_png_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    filename = resp.json()["filename"]
    yield filename
    import main  # noqa: PLC0415
    (main.UPLOAD_DIR / filename).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dataset directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset_dir(tmp_path, sample_png_bytes):
    """Temp dir with 2 valid PNGs + 1 .txt (simulates an image dataset)."""
    (tmp_path / "img1.png").write_bytes(sample_png_bytes)
    (tmp_path / "img2.png").write_bytes(sample_png_bytes)
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


# ---------------------------------------------------------------------------
# Patch torchvision read_image for detect tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_read_image(mocker):
    """Prevent real torchvision usage in detect_objects by mocking read_image."""
    return mocker.patch(
        "torchvision.io.image.read_image",
        return_value=MagicMock(),
    )
