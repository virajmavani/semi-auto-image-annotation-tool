# Architecture Documentation

## System Overview

Anno-Mage Web follows a modern client-server architecture with clear separation of concerns.

```
┌─────────────────────────────────────────────────────────────┐
│                        User's Browser                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              React Frontend (Port 3000)                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │  │
│  │  │   App    │  │Components│  │  AnnotationCanvas    │ │  │
│  │  │  (Main)  │──│  (UI)    │──│   (Konva.js)         │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────────┘ │  │
│  │       │                                                 │  │
│  │       │ HTTP/JSON                                       │  │
│  │       ▼                                                 │  │
│  │  ┌──────────┐                                          │  │
│  │  │API Client│                                          │  │
│  │  │ (Axios)  │                                          │  │
│  │  └──────────┘                                          │  │
│  └───────┬───────────────────────────────────────────────┘  │
└──────────┼──────────────────────────────────────────────────┘
           │
           │ REST API
           │
┌──────────▼──────────────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  API Endpoints                         │  │
│  │  /api/labels  /api/upload  /api/detect  /api/save     │  │
│  └───────┬────────────────────┬──────────────────────────┘  │
│          │                    │                              │
│  ┌───────▼──────────┐  ┌──────▼─────────────────────────┐  │
│  │  File Manager    │  │    Model Inference Engine      │  │
│  │  (Upload/Save)   │  │   (PyTorch + RetinaNet)        │  │
│  └──────────────────┘  └────────────────────────────────┘  │
│          │                    │                              │
│  ┌───────▼────────────────────▼──────────────────────────┐  │
│  │              File System Storage                       │  │
│  │    uploads/         annotations/        models/        │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Frontend Architecture

### Technology Stack
- **React 18**: Component-based UI
- **TypeScript**: Static typing for reliability
- **Vite**: Fast build tool and dev server
- **TailwindCSS**: Utility-first styling
- **Konva.js**: Canvas rendering engine
- **Axios**: HTTP client

### Component Hierarchy

```
App (Main Container)
│
├── Header
│   └── Title & Branding
│
├── Message Notification (Floating)
│   └── Success/Error Toast
│
└── Main Layout (Flex Container)
    │
    ├── Left Sidebar (Controls)
    │   ├── Image Upload
    │   ├── Threshold Slider
    │   ├── Auto Detect Button
    │   ├── Label Selector
    │   ├── Custom Label Input
    │   ├── Current Label Dropdown
    │   └── Action Buttons (Save/Clear)
    │
    ├── Center Canvas
    │   └── AnnotationCanvas
    │       ├── Konva Stage
    │       ├── Image Layer
    │       ├── Bounding Boxes
    │       ├── Corner Handles
    │       └── Draw Preview
    │
    └── Right Sidebar (Annotations)
        └── BBox List
            └── BBox Items (with delete)
```

### State Management

```typescript
// Main App State
{
  currentImage: ImageInfo | null,
  bboxes: BoundingBox[],
  labels: string[],
  selectedLabels: Set<string>,
  currentLabel: string | null,
  customLabel: string,
  threshold: number,
  isDetecting: boolean,
  message: { type, text } | null
}

// Types
interface BoundingBox {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  score?: number;
  color: string;
}

interface ImageInfo {
  filename: string;
  path: string;
  width: number;
  height: number;
}
```

### Data Flow

1. **Image Upload Flow**
   ```
   User selects file → handleImageUpload() → api.uploadImage()
   → Backend saves file → Returns ImageInfo → Update state
   ```

2. **Detection Flow**
   ```
   User clicks Detect → handleDetect() → api.detectObjects()
   → Backend runs model → Returns detections → Convert to BBoxes
   → Append to state
   ```

3. **Manual Drawing Flow**
   ```
   User drags on canvas → handleMouseDown/Move/Up
   → Calculate coordinates → onBboxCreate callback
   → Add to bboxes state
   ```

4. **Save Flow**
   ```
   User clicks Save → handleSave() → api.saveAnnotations()
   → Backend writes CSV & XML → Success message
   ```

## Backend Architecture

### Technology Stack
- **FastAPI**: Modern async web framework
- **PyTorch**: Deep learning inference
- **Torchvision**: Pre-trained models
- **Pillow**: Image processing
- **Pascal VOC Writer**: XML generation
- **Uvicorn**: ASGI server

### API Structure

```python
FastAPI App
│
├── Middleware
│   └── CORS (allow frontend origins)
│
├── Startup Event
│   └── Load ML Model
│
├── Endpoints
│   ├── GET  /              (Health check)
│   ├── GET  /api/labels    (Get COCO labels)
│   ├── POST /api/upload    (Upload image)
│   ├── GET  /api/image/:id (Serve image)
│   ├── POST /api/detect    (Run detection)
│   ├── POST /api/save      (Save annotations)
│   ├── POST /api/model/change (Change model/threshold)
│   └── GET  /api/images    (List images)
│
└── Storage
    ├── uploads/            (Uploaded images)
    └── annotations/        (CSV & VOC XML)
```

### Request/Response Flow

**Detection Endpoint Example:**
```
POST /api/detect
Request: {
  image_path: "/path/to/image.jpg",
  selected_labels: ["person", "car"]
}

Processing:
1. Read image with torchvision
2. Preprocess for model
3. Run inference
4. Filter by selected labels
5. Convert to response format

Response: {
  detections: [
    {
      x1: 100, y1: 200, x2: 300, y2: 400,
      label: "person",
      score: 0.95
    },
    ...
  ]
}
```

### Model Integration

```python
# Model Factory Pattern
ModelFactory.create_model(model_type, threshold)
  │
  ├── RetinaNetModel (current)
  │   ├── load_model()
  │   ├── preprocess_image()
  │   ├── predict()
  │   └── get_labels()
  │
  └── [Future models can be added here]
```

## File System Organization

```
web/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── uploads/             # Uploaded images (gitignored)
│   ├── annotations/         # Output (gitignored)
│   │   ├── annotations.csv
│   │   └── annotations_voc/
│   │       └── *.xml
│   └── venv/                # Virtual environment (gitignored)
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   └── AnnotationCanvas.tsx
    │   ├── api/
    │   │   └── client.ts
    │   ├── utils/
    │   │   └── colors.ts
    │   ├── types.ts
    │   ├── App.tsx
    │   ├── main.tsx
    │   └── index.css
    ├── public/
    ├── index.html
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    ├── tsconfig.json
    └── node_modules/        # (gitignored)
```

## Communication Protocol

### HTTP REST API

All communication uses JSON over HTTP:

```
Frontend                    Backend
   │                          │
   │  POST /api/upload        │
   │  multipart/form-data     │
   │─────────────────────────>│
   │                          │ Save file
   │  { filename, path,       │ Get dimensions
   │    width, height }       │
   │<─────────────────────────│
   │                          │
   │  POST /api/detect        │
   │  { image_path,           │
   │    selected_labels }     │
   │─────────────────────────>│
   │                          │ Load image
   │                          │ Run model
   │                          │ Filter results
   │  { detections: [...] }   │
   │<─────────────────────────│
```

## Canvas Rendering Architecture

### Konva.js Layer System

```
Stage (Container)
└── Layer
    ├── Image (Background)
    ├── Rectangles (Bounding Boxes)
    │   └── Props: { x, y, width, height, stroke, strokeWidth }
    ├── Text Labels (Above each box)
    │   └── Props: { x, y, text, fontSize, fill }
    └── Circles (Corner Handles)
        └── Props: { x, y, radius, fill, draggable }
```

### Event Handling

```
Canvas Events:
│
├── onMouseDown
│   ├── Check if clicking handle → Start edit mode
│   └── Else → Start drawing new box
│
├── onMouseMove
│   ├── Update crosshair guides
│   ├── Update zoom preview
│   └── If drawing → Update box preview
│
└── onMouseUp
    ├── If drawing → Create new bbox
    ├── If editing → Update bbox coordinates
    └── Clear temporary state
```

### Coordinate System

```
Canvas Coordinates (Display):
- Scaled to fit screen
- Example: 900x600 canvas

Original Image Coordinates:
- Actual image dimensions
- Example: 1920x1080 image

Conversion:
scale = min(canvasWidth / imageWidth, canvasHeight / imageHeight)
displayX = originalX * scale
originalX = displayX / scale
```

## Security Considerations

### Current Implementation
- **CORS**: Restricted to localhost origins
- **File Upload**: Saves to local directory
- **No Authentication**: Suitable for local use

### Production Considerations
- Add user authentication
- Implement file size limits
- Validate file types server-side
- Add rate limiting
- Use cloud storage (S3, etc.)
- Implement HTTPS
- Add CSRF protection

## Performance Optimizations

### Frontend
- **React.memo**: Memoize expensive components
- **useCallback**: Prevent unnecessary re-renders
- **Canvas rendering**: Hardware-accelerated via Konva
- **Image lazy loading**: Only load when needed

### Backend
- **Model caching**: Load model once on startup
- **Async operations**: Non-blocking I/O
- **Batch processing**: Future enhancement

## Deployment Options

### Development
```
Backend:  python main.py (localhost:8000)
Frontend: npm run dev (localhost:3000)
```

### Production Options

1. **Docker Compose**
   ```yaml
   services:
     backend:
       build: ./backend
       ports: ["8000:8000"]
     frontend:
       build: ./frontend
       ports: ["80:80"]
   ```

2. **Cloud Deployment**
   - Backend: AWS Lambda, Google Cloud Run
   - Frontend: Vercel, Netlify, S3 + CloudFront
   - Storage: S3, Google Cloud Storage

3. **Traditional Server**
   - Backend: Nginx + Gunicorn
   - Frontend: Static build served by Nginx

## Extensibility

### Adding New Models

1. Create model class extending `AbstractModel`:
```python
class YOLOModel(AbstractModel):
    def load_model(self, weights_path):
        # Implementation

    def preprocess_image(self, image):
        # Implementation

    def predict(self, preprocessed):
        # Implementation
```

2. Register in `ModelFactory`:
```python
if model_type.lower() == 'yolo':
    return YOLOModel(threshold=threshold)
```

3. Update frontend model selector

### Adding Export Formats

1. Create export function in backend:
```python
def export_to_yolo(bboxes, image_width, image_height):
    # Convert to YOLO format
    pass
```

2. Add endpoint:
```python
@app.post("/api/export/yolo")
async def export_yolo(request):
    # Implementation
```

3. Add UI button in frontend

## Error Handling

### Frontend
```typescript
try {
  await api.detectObjects(...)
} catch (error) {
  showMessage('error', 'Detection failed')
  console.error(error)
}
```

### Backend
```python
@app.post("/api/detect")
async def detect_objects(request):
    try:
        # Process
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    except Exception as e:
        raise HTTPException(500, f"Detection error: {str(e)}")
```

## Testing Strategy

### Frontend Testing
- Unit tests: Component logic with Jest
- Integration tests: API client with Mock Service Worker
- E2E tests: User flows with Playwright

### Backend Testing
- Unit tests: Model inference with pytest
- Integration tests: API endpoints with TestClient
- Load tests: Concurrent requests with Locust

## Monitoring & Logging

### Development
- Console logs in frontend
- Print statements in backend

### Production
- Frontend: Sentry for error tracking
- Backend: Structured logging with Python logging
- Metrics: Prometheus + Grafana
- Tracing: OpenTelemetry
