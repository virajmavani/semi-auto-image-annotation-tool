"""
FastAPI backend for Semi-Automatic Image Annotation Tool
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
import os
import asyncio
import csv
import json
from pathlib import Path
import base64
import io
from PIL import Image
import shutil

try:
    from models import ModelFactory
except ImportError:
    parent_dir = Path(__file__).parent.parent.parent.absolute()
    sys.path.insert(0, str(parent_dir))
    from models import ModelFactory

app = FastAPI(title="Anno-Mage API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
current_model = None
current_model_type = "owlv2"
current_threshold = 0.5
_model_lock = asyncio.Lock()

# Dataset directory
current_dataset_dir = None

# Upload size limit (50 MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# Paths
_DATA_DIR = Path.home() / ".anno-mage"
UPLOAD_DIR = _DATA_DIR / "uploads"
ANNOTATIONS_DIR = Path.cwd() / "annotations"
SNAPSHOTS_DIR = Path(os.environ.get("ANNO_MAGE_SNAPSHOTS", str(Path.cwd() / "snapshots")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Available models configuration
AVAILABLE_MODELS = {
    "retinanet": {
        "name": "RetinaNet ResNet50 FPN V2",
        "description": "Default PyTorch RetinaNet model (COCO pre-trained)",
        "type": "retinanet",
        "weights_path": None,
        "framework": "pytorch",
        "is_zero_shot": False,
    },
    "owlv2": {
        "name": "OWL-v2 (Zero-Shot)",
        "description": "Open-vocabulary — type any label",
        "type": "owlv2",
        "weights_path": None,
        "framework": "pytorch",
        "is_zero_shot": True,
    },
}


def scan_custom_models():
    """Scan snapshots directory for custom models"""
    if not SNAPSHOTS_DIR.exists():
        return

    # Scan for Keras models
    keras_dir = SNAPSHOTS_DIR / "keras"
    if keras_dir.exists():
        for model_file in keras_dir.glob("*.h5"):
            model_id = f"keras_{model_file.stem}"
            AVAILABLE_MODELS[model_id] = {
                "name": f"Custom Keras: {model_file.stem}",
                "description": f"Keras model from {model_file.name}",
                "type": "keras",
                "weights_path": str(model_file),
                "framework": "keras"
            }

    # Scan for TensorFlow models
    tf_dir = SNAPSHOTS_DIR / "tensorflow"
    if tf_dir.exists():
        for model_dir in tf_dir.iterdir():
            if model_dir.is_dir():
                frozen_graph = model_dir / "frozen_inference_graph.pb"
                if frozen_graph.exists():
                    model_id = f"tf_{model_dir.name}"
                    AVAILABLE_MODELS[model_id] = {
                        "name": f"Custom TensorFlow: {model_dir.name}",
                        "description": f"TensorFlow model from {model_dir.name}",
                        "type": "tensorflow",
                        "weights_path": str(frozen_graph),
                        "framework": "tensorflow"
                    }


def validate_filename(filename: str) -> None:
    """Validate filename to prevent path traversal attacks"""
    if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
        raise HTTPException(status_code=400, detail="Invalid filename")


def get_safe_image_path(filename: str) -> Path:
    """Get validated image path from dataset directory"""
    if current_dataset_dir is None:
        raise HTTPException(status_code=400, detail="No dataset directory set")

    validate_filename(filename)
    full_path = (current_dataset_dir / filename).resolve()

    # Ensure resolved path is still within dataset directory
    if not str(full_path).startswith(str(current_dataset_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return full_path


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: Optional[float] = None


class DetectionRequest(BaseModel):
    image_path: str
    selected_labels: List[str]


class SaveAnnotationRequest(BaseModel):
    image_name: str
    bboxes: List[BoundingBox]
    image_width: int
    image_height: int


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global current_model
    try:
        # Scan for custom models
        scan_custom_models()
        print(f"Found {len(AVAILABLE_MODELS)} available models")

        # Load default model
        current_model = ModelFactory.create_model(current_model_type, threshold=current_threshold)
        print(f"Model {current_model_type} loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")



@app.get("/api/models")
async def get_models():
    """Get list of available models"""
    models_list = []
    for model_id, model_info in AVAILABLE_MODELS.items():
        models_list.append({
            "id": model_id,
            "name": model_info["name"],
            "description": model_info["description"],
            "framework": model_info["framework"],
            "is_current": model_id == current_model_type,
            "is_zero_shot": model_info.get("is_zero_shot", False),
        })
    return {
        "models": models_list,
        "current_model": current_model_type,
        "current_threshold": current_threshold
    }


@app.get("/api/labels")
async def get_labels():
    """Get available COCO labels"""
    if current_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    labels = current_model.get_labels()
    return {"labels": labels}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    try:
        filename = Path(file.filename).name  # strip any directory components
        validate_filename(filename)
        if file.size and file.size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
        # Save uploaded file
        file_path = UPLOAD_DIR / filename
        size_written = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size_written += len(chunk)
                if size_written > MAX_UPLOAD_SIZE:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
                buffer.write(chunk)

        # Get image dimensions
        with Image.open(file_path) as img:
            width, height = img.size

        return {
            "filename": filename,
            "path": str(file_path),
            "width": width,
            "height": height
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Get uploaded image"""
    validate_filename(filename)
    file_path = (UPLOAD_DIR / filename).resolve()
    if not file_path.is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.post("/api/detect")
async def detect_objects(request: DetectionRequest):
    """Run object detection on an image"""
    async with _model_lock:
        model = current_model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        from torchvision.io.image import read_image

        # Validate image path is inside an allowed directory
        img_path = Path(request.image_path).resolve()
        allowed_dirs = [UPLOAD_DIR.resolve()]
        if current_dataset_dir:
            allowed_dirs.append(Path(current_dataset_dir).resolve())
        if not any(img_path.is_relative_to(d) for d in allowed_dirs):
            raise HTTPException(status_code=400, detail="Image path is outside allowed directories")

        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        img = read_image(str(img_path))

        # Preprocess and predict
        preprocessed = model.preprocess_image(img)

        def format_detection(d):
            return {
                "x1": int(d['box'][0]),
                "y1": int(d['box'][1]),
                "x2": int(d['box'][2]),
                "y2": int(d['box'][3]),
                "label": d['label'],
                "score": float(d['score']),
            }

        if model.is_zero_shot():
            if not request.selected_labels:
                return {"detections": []}
            detections = model.predict(preprocessed, text_queries=list(request.selected_labels))
            filtered_detections = [format_detection(d) for d in detections]
        else:
            detections = model.predict(preprocessed)
            filtered_detections = [
                format_detection(d) for d in detections
                if d['label'] in request.selected_labels
            ]

        return {"detections": filtered_detections}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/api/save")
async def save_annotations(request: SaveAnnotationRequest):
    """Save annotations to CSV and VOC XML format"""
    try:
        from pascal_voc_writer import Writer

        # Save to CSV (use csv.writer to prevent CSV injection)
        csv_path = ANNOTATIONS_DIR / "annotations.csv"
        with open(csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            for bbox in request.bboxes:
                writer_csv.writerow([
                    request.image_name, bbox.x1, bbox.y1,
                    bbox.x2, bbox.y2, bbox.label
                ])

        # Save to VOC XML
        voc_dir = ANNOTATIONS_DIR / "annotations_voc"
        voc_dir.mkdir(exist_ok=True)

        base_name = Path(request.image_name).stem
        writer = Writer(request.image_name, request.image_width, request.image_height)

        for bbox in request.bboxes:
            writer.addObject(bbox.label, bbox.x1, bbox.y1, bbox.x2, bbox.y2)

        xml_path = voc_dir / f"{base_name}.xml"
        writer.save(str(xml_path))

        # Save to JSONL (Hugging Face-compatible format)
        jsonl_path = ANNOTATIONS_DIR / "annotations.jsonl"
        record = {
            "image": request.image_name,
            "width": request.image_width,
            "height": request.image_height,
            "annotations": [
                {
                    "label": bbox.label,
                    "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                    **({"score": bbox.score} if bbox.score is not None else {}),
                }
                for bbox in request.bboxes
            ],
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return {
            "success": True,
            "csv_path": str(csv_path),
            "xml_path": str(xml_path),
            "jsonl_path": str(jsonl_path),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {str(e)}")


@app.post("/api/model/change")
async def change_model(model_id: str = Form(...), threshold: float = Form(0.5)):
    """Change the current model or threshold"""
    global current_model, current_model_type, current_threshold

    try:
        # Validate model exists
        if model_id not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        model_info = AVAILABLE_MODELS[model_id]

        # Load the appropriate model based on framework
        if model_info["framework"] == "pytorch":
            new_model = ModelFactory.create_model(
                model_info["type"],
                threshold=threshold,
                weights_path=model_info["weights_path"]
            )
        else:
            # For Keras/TensorFlow models, we'd need additional implementations
            # For now, only support PyTorch
            raise HTTPException(
                status_code=501,
                detail=f"{model_info['framework']} models not yet supported in web version. Only PyTorch RetinaNet is currently available."
            )

        # Update global state under lock
        async with _model_lock:
            current_model = new_model
            current_model_type = model_id
            current_threshold = threshold

        return {
            "success": True,
            "model_id": model_id,
            "model_name": model_info["name"],
            "threshold": threshold
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model change error: {str(e)}")


@app.get("/api/images")
async def list_images():
    """List all uploaded images"""
    try:
        images = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                with Image.open(file_path) as img:
                    width, height = img.size
                images.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "width": width,
                    "height": height
                })
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing images: {str(e)}")


@app.post("/api/dataset/set")
async def set_dataset_directory(directory_path: str = Form(...)):
    """Set the working directory containing images to annotate"""
    global current_dataset_dir

    try:
        dataset_path = Path(directory_path).expanduser().resolve()

        # Validate directory exists
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")

        if not dataset_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")

        # Scan for images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        images = []

        for file_path in sorted(dataset_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                    images.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "width": width,
                        "height": height
                    })
                except Exception:
                    # Skip files that can't be opened as images
                    continue

        if not images:
            raise HTTPException(status_code=400, detail="No valid images found in directory")

        # Set the current dataset directory
        current_dataset_dir = dataset_path

        return {
            "success": True,
            "directory": str(dataset_path),
            "image_count": len(images),
            "images": images
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting dataset directory: {str(e)}")


@app.get("/api/dataset/images")
async def list_dataset_images():
    """List all images in the current dataset directory"""
    if current_dataset_dir is None:
        raise HTTPException(status_code=400, detail="No dataset directory set")

    try:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        images = []

        for file_path in sorted(current_dataset_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                    images.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "width": width,
                        "height": height
                    })
                except Exception:
                    continue

        return {"images": images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing dataset images: {str(e)}")


@app.get("/api/dataset/image/{filename}")
async def get_dataset_image(filename: str):
    """Serve image from dataset directory"""
    image_path = get_safe_image_path(filename)
    return FileResponse(image_path)


@app.get("/api/browse")
async def browse_directory(path: str = "~"):
    """List subdirectories at the given path for the directory browser UI"""
    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.is_relative_to(Path.home()):
            raise HTTPException(status_code=400, detail="Path is outside the allowed directory")

        if not dir_path.exists() or not dir_path.is_dir():
            raise HTTPException(status_code=400, detail="Invalid directory path")

        entries = []
        try:
            for entry in sorted(dir_path.iterdir()):
                if entry.is_dir() and not entry.name.startswith('.'):
                    entries.append({
                        "name": entry.name,
                        "path": str(entry),
                    })
        except PermissionError:
            pass  # Return what we have, skip unreadable entries

        parent = dir_path.parent
        parent_path = str(parent) if parent != dir_path else None

        return {
            "current_path": str(dir_path),
            "parent_path": parent_path,
            "entries": entries,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/annotations/{filename}")
async def load_annotations(filename: str):
    """Load previously saved annotations for an image"""
    try:
        validate_filename(filename)

        # Try to load from CSV first
        csv_path = ANNOTATIONS_DIR / "annotations.csv"
        bboxes = []

        if csv_path.exists():
            with open(csv_path, "r") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6 and parts[0] == filename:
                        bboxes.append({
                            "x1": int(parts[1]),
                            "y1": int(parts[2]),
                            "x2": int(parts[3]),
                            "y2": int(parts[4]),
                            "label": parts[5]
                        })

        return {"bboxes": bboxes}

    except Exception as e:
        # Return empty if no annotations exist
        return {"bboxes": []}


from importlib.resources import files as pkg_files
try:
    _static = pkg_files("anno_mage").joinpath("static")
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
except Exception:
    pass  # Running in dev mode without bundled frontend


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
