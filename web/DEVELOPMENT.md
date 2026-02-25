# Development Guide

Guide for developers who want to contribute to or modify Anno-Mage Web.

## Development Environment Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn
- Git
- Code editor (VS Code recommended)

### Recommended VS Code Extensions
- Python
- Pylance
- ESLint
- Prettier
- Tailwind CSS IntelliSense
- ES7+ React/Redux snippets

### Initial Setup

```bash
# Clone the repository
git clone <repo-url>
cd semi-auto-image-annotation-tool/web

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision

# Frontend setup
cd ../frontend
npm install

# Return to web directory
cd ..
```

## Development Workflow

### Running in Development Mode

**Terminal 1 - Backend with auto-reload:**
```bash
cd web/backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend with HMR:**
```bash
cd web/frontend
npm run dev
```

### Code Style

#### Python (Backend)
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for functions

```python
def detect_objects(
    image_path: str,
    selected_labels: List[str],
    threshold: float = 0.5
) -> List[Detection]:
    """
    Run object detection on an image.

    Args:
        image_path: Path to the image file
        selected_labels: List of label names to filter
        threshold: Detection confidence threshold

    Returns:
        List of Detection objects
    """
    # Implementation
```

#### TypeScript (Frontend)
- Use TypeScript strictly (no `any` types)
- Functional components with hooks
- Props interfaces for all components
- Maximum line length: 100 characters

```typescript
interface Props {
  imageUrl: string;
  onBboxCreate: (bbox: Omit<BoundingBox, 'id'>) => void;
}

export const AnnotationCanvas: React.FC<Props> = ({
  imageUrl,
  onBboxCreate
}) => {
  // Implementation
};
```

### Project Structure Conventions

#### Backend
```
backend/
├── main.py              # FastAPI app and endpoints
├── models/              # ML model wrappers (if needed locally)
├── utils/               # Helper functions
├── schemas/             # Pydantic models (future)
└── tests/               # Backend tests (future)
```

#### Frontend
```
frontend/src/
├── components/          # React components
│   └── [ComponentName].tsx
├── api/                 # API client
│   └── client.ts
├── utils/               # Utility functions
│   └── [utility].ts
├── types.ts             # TypeScript interfaces
├── App.tsx              # Main app component
└── main.tsx             # Entry point
```

## Adding New Features

### Backend Feature

1. **Define the endpoint:**
```python
@app.post("/api/new-feature")
async def new_feature(request: RequestModel):
    """Endpoint description"""
    try:
        result = process_feature(request)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
```

2. **Add request/response models:**
```python
class RequestModel(BaseModel):
    param1: str
    param2: int

class ResponseModel(BaseModel):
    success: bool
    data: Any
```

3. **Test the endpoint:**
```bash
# Visit http://localhost:8000/docs
# Use Swagger UI to test
```

### Frontend Feature

1. **Add API client method:**
```typescript
// api/client.ts
export const api = {
  async newFeature(param1: string, param2: number): Promise<Result> {
    const response = await axios.post(
      `${API_BASE_URL}/api/new-feature`,
      { param1, param2 }
    );
    return response.data;
  }
};
```

2. **Create component (if needed):**
```typescript
// components/NewComponent.tsx
import React from 'react';

interface NewComponentProps {
  data: string;
}

export const NewComponent: React.FC<NewComponentProps> = ({ data }) => {
  return <div>{data}</div>;
};
```

3. **Integrate into App:**
```typescript
// App.tsx
import { NewComponent } from './components/NewComponent';

// In App component:
const [featureData, setFeatureData] = useState<string>('');

const handleFeature = async () => {
  const result = await api.newFeature('param', 42);
  setFeatureData(result.data);
};

// In JSX:
<NewComponent data={featureData} />
```

## Common Development Tasks

### Adding a New COCO Label Category

Labels are loaded from the model, but you can filter or add custom ones:

**Frontend:**
```typescript
// In App.tsx
const handleAddCustomLabel = () => {
  if (customLabel && !labels.includes(customLabel)) {
    setLabels([...labels, customLabel]);
  }
};
```

### Changing the Color Palette

**Edit `frontend/src/utils/colors.ts`:**
```typescript
export const COLORS = [
  '#your-color-1',
  '#your-color-2',
  // ...
];
```

### Adjusting Detection Threshold Default

**Backend (`backend/main.py`):**
```python
current_threshold = 0.7  # Change from 0.5
```

**Frontend (`frontend/src/App.tsx`):**
```typescript
const [threshold, setThreshold] = useState(0.7);  // Change from 0.5
```

### Adding Export Format

**Backend:**
```python
@app.post("/api/export/custom-format")
async def export_custom_format(request: SaveAnnotationRequest):
    # Convert annotations to custom format
    with open('output.custom', 'w') as f:
        for bbox in request.bboxes:
            f.write(f"{bbox.label} {bbox.x1} {bbox.y1} {bbox.x2} {bbox.y2}\n")
    return {"success": True}
```

**Frontend:**
```typescript
// Add export button
<button onClick={handleExportCustom}>
  Export Custom Format
</button>

// Add handler
const handleExportCustom = async () => {
  await api.exportCustomFormat(imageName, bboxes, width, height);
};
```

## Debugging

### Backend Debugging

**Add debug logs:**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.post("/api/detect")
async def detect_objects(request):
    logger.debug(f"Detection request: {request.dict()}")
    # ...
```

**Use Python debugger:**
```python
import pdb; pdb.set_trace()
```

**Check logs:**
```bash
# Backend terminal shows all logs
```

### Frontend Debugging

**Browser DevTools:**
- Console: Check errors and logs
- Network: Inspect API calls
- React DevTools: Component state

**Add debug logs:**
```typescript
console.log('State:', { bboxes, currentImage });
console.error('Error occurred:', error);
```

**React DevTools:**
Install the browser extension to inspect component props and state.

## Testing

### Backend Tests (pytest)

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_labels():
    response = client.get("/api/labels")
    assert response.status_code == 200
    assert "labels" in response.json()
```

**Run tests:**
```bash
cd backend
pytest
```

### Frontend Tests (Jest + React Testing Library)

```typescript
// components/__tests__/AnnotationCanvas.test.tsx
import { render, screen } from '@testing-library/react';
import { AnnotationCanvas } from '../AnnotationCanvas';

test('renders canvas', () => {
  render(<AnnotationCanvas {...props} />);
  const canvas = screen.getByRole('img');
  expect(canvas).toBeInTheDocument();
});
```

**Run tests:**
```bash
cd frontend
npm test
```

## Performance Optimization

### Backend

**Profile endpoint performance:**
```python
import time

@app.post("/api/detect")
async def detect_objects(request):
    start = time.time()
    result = model.predict(...)
    print(f"Detection took {time.time() - start:.2f}s")
    return result
```

**Optimize model inference:**
- Use GPU if available
- Batch predictions
- Cache model in memory

### Frontend

**Optimize re-renders:**
```typescript
// Memoize expensive components
const AnnotationCanvas = React.memo(({ ... }) => {
  // Component logic
});

// Memoize callbacks
const handleBboxCreate = useCallback((bbox) => {
  // Logic
}, [dependencies]);
```

**Lazy load components:**
```typescript
const HeavyComponent = React.lazy(() => import('./HeavyComponent'));

<Suspense fallback={<div>Loading...</div>}>
  <HeavyComponent />
</Suspense>
```

## Git Workflow

### Branch Naming
- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `refactor/component-name` - Code refactoring
- `docs/update-readme` - Documentation

### Commit Messages
Follow conventional commits:
```
feat: add batch image upload
fix: resolve canvas scaling issue
docs: update installation guide
refactor: simplify API client
style: format code with prettier
```

### Pull Request Process

1. Create feature branch
2. Make changes
3. Test thoroughly
4. Commit with clear messages
5. Push and create PR
6. Wait for review
7. Address feedback
8. Merge when approved

## Troubleshooting Development Issues

### Backend port already in use
```bash
lsof -ti:8000 | xargs kill -9
```

### Frontend build errors
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### CORS issues
Check CORS middleware in `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your origin
    ...
)
```

### Model not loading
- Verify PyTorch installation
- Check model imports
- Ensure parent `models/` directory is accessible

### TypeScript errors
```bash
cd frontend
npx tsc --noEmit  # Check all type errors
```

## Useful Commands

### Backend
```bash
# Format code
black main.py

# Check types
mypy main.py

# Run linter
flake8 main.py

# Install new package
pip install package-name
pip freeze > requirements.txt
```

### Frontend
```bash
# Format code
npm run lint

# Type check
npm run type-check

# Build for production
npm run build

# Preview production build
npm run preview

# Install new package
npm install package-name
```

## Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [TailwindCSS Docs](https://tailwindcss.com/docs)
- [Konva.js Docs](https://konvajs.org/docs/)

### Learning Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [React Hooks](https://react.dev/reference/react)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

## Getting Help

1. Check existing issues on GitHub
2. Read documentation
3. Ask in discussions
4. Create a new issue with details:
   - What you tried
   - Expected behavior
   - Actual behavior
   - Error messages
   - Environment details
