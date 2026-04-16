# Anno-Mage: A Semi Automatic Image Annotation Tool

[![PyPI version](https://img.shields.io/pypi/v/anno-mage)](https://pypi.org/project/anno-mage/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anno-mage)](https://pypi.org/project/anno-mage/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/anno-mage)](https://pypi.org/project/anno-mage/)
[![Publish to PyPI](https://github.com/virajmavani/semi-auto-image-annotation-tool/actions/workflows/release.yml/badge.svg)](https://github.com/virajmavani/semi-auto-image-annotation-tool/actions/workflows/release.yml)
[![License](https://img.shields.io/github/license/virajmavani/semi-auto-image-annotation-tool)](LICENSE)

![Demo](https://raw.githubusercontent.com/virajmavani/semi-auto-image-annotation-tool/master/demo.gif)

Semi-automatic image annotation toolbox powered by PyTorch object detection models. Available as both a **desktop app** (Tkinter) and a **web app** (FastAPI + React).

## Interfaces

### Desktop App
A Tkinter GUI that runs locally. Annotate images from a directory with keyboard navigation and a precision zoom panel.

### Web App
A browser-based interface with a modern dark/light UI, REST API, and dataset directory browsing. See [`web/README.md`](web/README.md) for full documentation.

---

## Desktop App

### Dependencies

- Python 3.12
- PyTorch + Torchvision (for RetinaNet inference)
- Pillow, pascal-voc-writer

Custom snapshots from legacy Keras (`.h5`) or TensorFlow (`frozen_inference_graph.pb`) models can be placed in `snapshots/keras/` and `snapshots/tensorflow/` respectively and will be listed in the model menu.

### Installation

1. Clone this repository.

2. Create and activate a virtual environment:
   ```bash
   python -m venv my_venv
   source my_venv/bin/activate  # On Windows: my_venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision pillow pascal-voc-writer
   ```

4. Run the app:
   ```bash
   python main.py
   ```

The default model (RetinaNet ResNet50 FPN V2) downloads its weights automatically from the PyTorch model hub on first run.

### Usage

1. Click **Open Dir** to select a folder of images, or **Open** for a single image.
2. Use **Add All Classes** to load all 80 COCO labels, or pick individual ones from the dropdown.
3. Select **Auto** suggestion mode and click **Detect** to run object detection.
4. To annotate manually, select a class from the list, then drag a bounding box on the canvas.
5. Use **← →** arrow keys to move between images. Annotations save automatically on navigation.
6. Final annotations are written to:
   - `annotations/annotations.csv`
   - `annotations/annotations_voc/<image>.xml` (Pascal VOC)

To annotate a custom (non-COCO) dataset, update the label map in `config.py` before running.

### Tested on

- Windows 10
- Ubuntu 16.04
- macOS High Sierra

---

## Web App

See [`web/README.md`](web/README.md) for installation, usage, and API reference.

**Quick start:**
```bash
# Backend (port 8000)
cd web/backend && python main.py

# Frontend (port 3000)
cd web/frontend && npm install && npm run dev
```

Or use the convenience script:
```bash
cd web && bash start.sh
```

---

## PyPI Distribution

### Install from PyPI

```bash
pip install anno-mage
anno-mage
```

The app opens in your browser automatically. Annotations are saved to `~/.anno-mage/annotations/`.

### Publish a Release

Releases publish automatically to PyPI when a version tag is pushed. GitHub Actions builds the frontend, packages everything, and publishes via PyPI Trusted Publishers (no tokens required).

**One-time PyPI setup:**
1. Go to your PyPI project → *Manage* → *Publishing* → *Add a new publisher*
2. Set: GitHub repo `virajmavani/semi-auto-image-annotation-tool`, workflow `release.yml`, environment `pypi`

**To release:**
```bash
git tag v2.0.1
git push origin v2.0.1
```

That's it — the workflow in `.github/workflows/release.yml` handles the rest.

### Build Locally

To build the package without publishing:

**Prerequisites:**
```bash
pip install build
npm install  # inside web/frontend if not already done
```

```bash
bash build_release.sh
```

This compiles the React frontend, copies the build into `anno_mage/static/`, and produces wheel and sdist artifacts in `dist/`.

---

## Output Formats

Both interfaces produce identical output:

| Format | Location | Description |
|--------|----------|-------------|
| CSV | `annotations/annotations.csv` | `image_path,x1,y1,x2,y2,label` per row |
| Pascal VOC XML | `annotations/annotations_voc/` | One XML file per image |

---

## Acknowledgments

- [Meditab Software Inc.](https://www.meditab.com/)
- [PyTorch / Torchvision](https://pytorch.org/) for the RetinaNet implementation
- [Computer Vision Group](https://cvgldce.github.io/), L.D. College of Engineering

### Join the developers channel

Slack: https://join.slack.com/t/annomage/shared_invite/zt-dh4ca9du-4VOcwUMCSNA6lmyG~tNUPg
