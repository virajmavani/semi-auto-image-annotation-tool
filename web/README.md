# Anno-Mage Web

A browser-based image annotation tool built with React, TypeScript, and FastAPI. Shares the same PyTorch-based detection model as the desktop app.

![React](https://img.shields.io/badge/React-18.2-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178c6)

## Features

- **Dataset mode**: Point to a local image directory, browse images in a grid, and navigate with ← → arrow keys
- **Directory browser**: A folder-picker modal lets you navigate the filesystem without typing paths
- **Auto-detection**: RetinaNet ResNet50 FPN V2 (COCO, 80 classes) with adjustable confidence threshold
- **Manual annotation**: Click-and-drag to draw bounding boxes; click a box to select, then drag corners or edges to resize
- **Auto-detect on load**: Optionally run detection automatically each time an image is opened
- **Multiple export formats**: CSV and Pascal VOC XML
- **Custom labels**: Add labels beyond the COCO set
- **Dark / light theme**: Toggle in the header
- **Keyboard shortcuts**: ← → to navigate images, Ctrl+S to save

## Installation

### Prerequisites

- Python 3.8+
- Node.js 18+
- PyTorch + Torchvision ([pytorch.org](https://pytorch.org) for system-specific instructions)

### Backend

```bash
cd web/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision
```

### Frontend

```bash
cd web/frontend
npm install
```

## Running

### Both servers at once (Linux / macOS)

```bash
cd web && bash start.sh
```

### Manual start

**Terminal 1 — Backend (port 8000):**
```bash
cd web/backend && python main.py
```

**Terminal 2 — Frontend (port 3000):**
```bash
cd web/frontend && npm run dev
```

| Service | URL |
|---------|-----|
| Web app | http://localhost:3000 |
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

## Usage Guide

### Dataset mode (recommended for batch annotation)

1. Click the **folder icon** next to "Dataset Directory" to open the directory browser, navigate to your image folder, and click **Select**. Or type the path directly.
2. Click **Load**. The right panel shows a thumbnail grid of all images found; the first image opens automatically.
3. Navigate with the **← Previous** / **Next →** buttons or the **← →** arrow keys. Annotations for the current image are saved automatically on navigation.
4. Enable **Auto-detect on image load** to run detection every time a new image opens.

### Single image mode

Click **Choose File** in the "Or Upload Single Image" section to upload a single image from your machine.

### Annotating

1. Select one or more labels from **Select Labels** (use **Select All** to check all 80 COCO classes).
2. Choose a **Current Label for Drawing** from the dropdown.
3. Click **Auto Detect** to run AI detection, or draw boxes manually by clicking and dragging on the canvas.
4. Click a box to select it — corner and edge handles appear for resizing.
5. Boxes can be deleted individually from the right-side annotations panel.
6. Click **Save** to write annotations, or press **Ctrl+S**.

### Threshold

Move the **Detection Threshold** slider before clicking Auto Detect. The model is reloaded at the new threshold in real time.

### Custom labels

Type a label name in **Add Custom Label** and press Enter or click **+**. Custom labels appear alongside COCO labels in the drawing dropdown.

## Annotation Output

| Format | Path |
|--------|------|
| CSV | `web/backend/annotations/annotations.csv` |
| Pascal VOC XML | `web/backend/annotations/annotations_voc/<image>.xml` |

CSV format: `image_name,x1,y1,x2,y2,label`

## API Reference

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List available models; auto-scans `snapshots/` |
| POST | `/api/model/change` | Change model or threshold (form-data: `model_id`, `threshold`) |

### Labels

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/labels` | COCO class names for the current model |

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload a single image (multipart) |
| GET | `/api/image/{filename}` | Serve an uploaded image |
| GET | `/api/images` | List all uploaded images |

### Dataset directory

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/dataset/set` | Set working directory (form-data: `directory_path`); returns image list |
| GET | `/api/dataset/images` | List images in current dataset directory |
| GET | `/api/dataset/image/{filename}` | Serve an image from the dataset directory |
| GET | `/api/browse?path=~` | List subdirectories at `path` for the directory browser UI |

### Annotations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Run detection (`{ image_path, selected_labels }`) |
| POST | `/api/save` | Save annotations to CSV + VOC XML |
| GET | `/api/annotations/{filename}` | Load previously saved annotations for an image |

## Project Structure

```
web/
├── backend/
│   ├── main.py              # FastAPI app and all endpoints
│   ├── requirements.txt
│   ├── uploads/             # Single-image uploads (gitignored)
│   └── annotations/         # Output (gitignored)
│       ├── annotations.csv
│       └── annotations_voc/
│
└── frontend/
    └── src/
        ├── components/
        │   ├── AnnotationCanvas.tsx   # Konva.js canvas; drawing, selection, resize, move
        │   └── DirectoryBrowser.tsx   # Filesystem navigator modal
        ├── api/
        │   └── client.ts              # Typed Axios wrappers for all endpoints
        ├── utils/
        │   └── colors.ts
        ├── types.ts
        └── App.tsx                    # All application state and layout
```

The backend imports the `models/` package from the repository root (two directories up), so the RetinaNet implementation is shared with the desktop app.

## Adding Custom Models

Place model files in the `snapshots/` directory at the repo root:

- **Keras** (`.h5`): `snapshots/keras/<name>.h5`
- **TensorFlow** (`frozen_inference_graph.pb`): `snapshots/tensorflow/<name>/frozen_inference_graph.pb`

The backend scans these on startup and lists them in the model dropdown. To make them fully functional for inference, implement a subclass of `AbstractModel` in `models/` and register it in `ModelFactory`. Currently only PyTorch RetinaNet is wired for inference.

## Troubleshooting

**Backend port 8000 in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Frontend port 3000 in use** — edit `web/frontend/vite.config.ts`:
```typescript
server: { port: 3001 }
```

**CORS errors** — ensure `allow_origins` in `web/backend/main.py` includes your frontend URL.

**Model not loading** — verify PyTorch is installed and the `models/` directory at the repo root is accessible from the backend working directory.

## License

Apache License 2.0 — see the root [`LICENSE`](../LICENSE) file.
