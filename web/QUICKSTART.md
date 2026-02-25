# Quick Start Guide

Get Anno-Mage Web running in 5 minutes.

## Setup (one-time)

### Backend
```bash
cd web/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision  # see pytorch.org for GPU-specific instructions
```

### Frontend
```bash
cd web/frontend
npm install
```

## Running

### Option A: Start script (Linux / macOS)
```bash
cd web && bash start.sh
```

### Option B: Manual

**Terminal 1 — Backend:**
```bash
cd web/backend && python main.py
```

**Terminal 2 — Frontend:**
```bash
cd web/frontend && npm run dev
```

Open **http://localhost:3000** in your browser.

- API docs: http://localhost:8000/docs

## First Annotation — Dataset Mode

1. Click the **folder icon** next to "Dataset Directory" and navigate to your image folder, or type the path directly.
2. Click **Load** — thumbnails appear in the right panel.
3. Check a few labels (e.g. "person", "car") in **Select Labels**.
4. Click **Auto Detect** to run AI detection on the current image.
5. To draw manually, pick a label in **Current Label for Drawing**, then drag on the canvas.
6. Press **Ctrl+S** or click **Save**. Use **← →** arrow keys to move to the next image.

## First Annotation — Single Image

1. Click **Choose File** under "Or Upload Single Image".
2. Check labels and click **Auto Detect**, or draw boxes manually.
3. Click **Save**.

## Output Files

Annotations are saved to:
- CSV: `web/backend/annotations/annotations.csv`
- Pascal VOC XML: `web/backend/annotations/annotations_voc/`

## Troubleshooting

**Port 8000 in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Port 3000 in use** — edit `web/frontend/vite.config.ts`:
```typescript
server: { port: 3001 }
```

**Model not loading** — make sure you're running from the repo root directory so the backend can find the `models/` package.

## Next Steps

See [README.md](README.md) for full documentation and API reference.
