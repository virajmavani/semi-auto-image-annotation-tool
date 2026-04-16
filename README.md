# Anno-Mage: A Semi Automatic Image Annotation Tool

[![PyPI version](https://img.shields.io/pypi/v/anno-mage)](https://pypi.org/project/anno-mage/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anno-mage)](https://pypi.org/project/anno-mage/)
[![Publish to PyPI](https://github.com/virajmavani/semi-auto-image-annotation-tool/actions/workflows/release.yml/badge.svg)](https://github.com/virajmavani/semi-auto-image-annotation-tool/actions/workflows/release.yml)
[![Tests](https://github.com/virajmavani/semi-auto-image-annotation-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/virajmavani/semi-auto-image-annotation-tool/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/virajmavani/semi-auto-image-annotation-tool)](LICENSE)

![Demo](https://raw.githubusercontent.com/virajmavani/semi-auto-image-annotation-tool/master/demo.gif)

Semi-automatic image annotation toolbox powered by PyTorch object detection models, including open-vocabulary zero-shot detection via OWL-v2. Available as a **web app** (FastAPI + React).

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
- [HuggingFace Transformers](https://huggingface.co/google/owlv2-base-patch16-ensemble) for the OWL-v2 zero-shot detection model
- [Computer Vision Group](https://cvgldce.github.io/), L.D. College of Engineering

### Join the developers channel

Slack: https://join.slack.com/t/annomage/shared_invite/zt-dh4ca9du-4VOcwUMCSNA6lmyG~tNUPg
