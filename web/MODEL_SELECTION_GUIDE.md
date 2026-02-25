# Model Selection Feature Guide

## Overview

The Anno-Mage Web application now supports **dynamic model selection**, allowing you to choose from multiple object detection models through the UI.

## Features

✅ **Automatic Model Discovery**: Scans `snapshots/` directory for custom models
✅ **Real-time Switching**: Change models without restarting the application
✅ **Model Information**: View model name, framework, and description
✅ **Framework Support**: Detects PyTorch, Keras, and TensorFlow models
✅ **Threshold Sync**: Threshold updates automatically when switching models

## How It Works

### Backend

The backend automatically scans the `snapshots/` directory on startup to discover available models:

```
snapshots/
├── keras/
│   └── model_name.h5          → Detected as "Custom Keras: model_name"
└── tensorflow/
    └── model_dir/
        └── frozen_inference_graph.pb  → Detected as "Custom TensorFlow: model_dir"
```

### Frontend

The UI displays a dropdown with all available models, showing:
- Model name
- Framework (pytorch, keras, tensorflow)
- Description

## Usage

### 1. Select a Model

In the left sidebar, you'll see a **"Detection Model"** dropdown:

```
Detection Model
┌──────────────────────────────────────────┐
│ RetinaNet ResNet50 FPN V2 (pytorch)  ▼  │
└──────────────────────────────────────────┘
Default PyTorch RetinaNet model (COCO pre-trained)
```

### 2. Switch Models

Simply select a different model from the dropdown. The app will:
1. Load the new model on the backend
2. Update the available labels
3. Clear selected labels
4. Show a success message

### 3. Adjust Threshold

The threshold slider works with the currently selected model. Changing it updates the model's detection confidence threshold in real-time.

## Supported Models

### ✅ Fully Supported

**PyTorch RetinaNet** (default)
- Framework: PyTorch
- Architecture: RetinaNet with ResNet50 FPN V2 backbone
- Pre-trained: MS COCO dataset (80 classes)
- Features: Real-time detection, adjustable threshold

### 🔍 Detected (Not Yet Implemented)

**Keras Models**
- Status: Detected and listed, but not yet functional
- File pattern: `snapshots/keras/*.h5`
- Note: Requires implementation of KerasModel class

**TensorFlow Models**
- Status: Detected and listed, but not yet functional
- File pattern: `snapshots/tensorflow/*/frozen_inference_graph.pb`
- Note: Requires implementation of TensorFlowModel class

## Adding Custom Models

### Option 1: Use Existing PyTorch Model

If you have a custom PyTorch model compatible with the RetinaNet interface:

1. Save your model weights
2. You can load it through the API (future enhancement)

### Option 2: Add Keras Model Support

To fully implement Keras model support:

1. **Create Keras model class** in `models/keras_model.py`:

```python
from .model_provider import AbstractModel
import tensorflow as tf

class KerasModel(AbstractModel):
    def load_model(self, weights_path: str = None):
        self.model = tf.keras.models.load_model(weights_path)
        # Set up preprocessing, labels, etc.

    def preprocess_image(self, image):
        # Implement preprocessing
        pass

    def predict(self, preprocessed_image):
        # Run inference and return standardized format
        pass

    def get_labels(self):
        # Return class labels
        pass
```

2. **Update ModelFactory** in `models/factory.py`:

```python
from .keras_model import KerasModel

def create_model(model_type: str, threshold: float = 0.5, weights_path: Optional[str] = None):
    if model_type.lower() == 'keras':
        model = KerasModel(threshold=threshold)
        model.load_model(weights_path)
        return model
    # ... existing code
```

3. **Update backend** `web/backend/main.py`:

```python
# In change_model function
if model_info["framework"] == "keras":
    current_model = ModelFactory.create_model(
        "keras",
        threshold=threshold,
        weights_path=model_info["weights_path"]
    )
```

### Option 3: Add TensorFlow Model Support

Similar process as Keras, but implement `TensorFlowModel` class.

## API Reference

### GET /api/models

Returns list of available models:

```json
{
  "models": [
    {
      "id": "retinanet",
      "name": "RetinaNet ResNet50 FPN V2",
      "description": "Default PyTorch RetinaNet model (COCO pre-trained)",
      "framework": "pytorch",
      "is_current": true
    },
    {
      "id": "keras_my_model",
      "name": "Custom Keras: my_model",
      "description": "Keras model from my_model.h5",
      "framework": "keras",
      "is_current": false
    }
  ],
  "current_model": "retinanet",
  "current_threshold": 0.5
}
```

### POST /api/model/change

Change the active model:

**Request:**
```
Content-Type: multipart/form-data

model_id=retinanet
threshold=0.7
```

**Response:**
```json
{
  "success": true,
  "model_id": "retinanet",
  "model_name": "RetinaNet ResNet50 FPN V2",
  "threshold": 0.7
}
```

**Error Response (unsupported framework):**
```json
{
  "detail": "keras models not yet supported in web version. Only PyTorch RetinaNet is currently available."
}
```

## Architecture

### Model Discovery Flow

```
Startup
   │
   ├─> scan_custom_models()
   │      │
   │      ├─> Scan snapshots/keras/*.h5
   │      │      └─> Add to AVAILABLE_MODELS dict
   │      │
   │      └─> Scan snapshots/tensorflow/*/frozen_inference_graph.pb
   │             └─> Add to AVAILABLE_MODELS dict
   │
   └─> Load default model (retinanet)
```

### Model Switch Flow

```
User selects model
   │
   ├─> Frontend: handleModelChange()
   │      └─> api.changeModel(modelId, threshold)
   │
   ├─> Backend: change_model()
   │      ├─> Validate model exists
   │      ├─> Check framework support
   │      ├─> Create model instance
   │      └─> Return success
   │
   └─> Frontend: Update UI
          ├─> Reload labels
          ├─> Clear selections
          └─> Show success message
```

## Configuration

### Backend Configuration

In `web/backend/main.py`:

```python
# Available models configuration
AVAILABLE_MODELS = {
    "retinanet": {
        "name": "RetinaNet ResNet50 FPN V2",
        "description": "Default PyTorch RetinaNet model (COCO pre-trained)",
        "type": "retinanet",
        "weights_path": None,
        "framework": "pytorch"
    }
}

# Custom models are added by scan_custom_models()
```

### Frontend Configuration

In `web/frontend/src/App.tsx`:

```typescript
const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
const [currentModelId, setCurrentModelId] = useState<string>('retinanet');

// Load models on mount
useEffect(() => {
  loadModelsAndLabels();
}, []);
```

## Troubleshooting

### Model not appearing in dropdown

**Problem:** Custom model file in `snapshots/` but not showing in UI

**Solutions:**
1. Check file location:
   - Keras: `snapshots/keras/yourmodel.h5`
   - TensorFlow: `snapshots/tensorflow/yourmodel/frozen_inference_graph.pb`
2. Restart backend server to rescan
3. Check backend console for scanning messages

### Error when selecting model

**Problem:** "Model not yet supported" error

**Cause:** Only PyTorch models are currently functional

**Solution:** Either:
- Use the default RetinaNet model
- Implement model class (see "Adding Custom Models" above)

### Labels don't update after model change

**Problem:** Old labels still showing

**Cause:** Frontend state not refreshing

**Solution:**
- Refresh the page
- Check browser console for errors
- Verify backend is responding correctly

## Best Practices

### Development

1. **Test model before adding**: Ensure your custom model works standalone
2. **Follow AbstractModel interface**: Implement all required methods
3. **Handle errors gracefully**: Return appropriate error messages
4. **Document your model**: Add clear description in AVAILABLE_MODELS

### Production

1. **Version your models**: Include version in model name
2. **Monitor performance**: Different models have different speeds
3. **Validate outputs**: Ensure model returns expected format
4. **Set appropriate thresholds**: Test and tune for your use case

## Future Enhancements

- [ ] Full Keras model support
- [ ] Full TensorFlow model support
- [ ] Model upload via UI
- [ ] Model performance metrics
- [ ] Model comparison mode
- [ ] Custom model configuration UI
- [ ] Model version management
- [ ] A/B testing between models

## Examples

### Example: Switching to a different threshold

```javascript
// User moves slider to 0.7
handleThresholdChange(0.7)
  ↓
api.changeModel('retinanet', 0.7)
  ↓
Backend updates model.box_score_thresh = 0.7
  ↓
Next detection uses new threshold
```

### Example: Adding a custom PyTorch model

```python
# In your model training script
model = create_your_custom_retinanet()
torch.save(model.state_dict(), 'custom_model.pth')

# In ModelFactory
AVAILABLE_MODELS["custom"] = {
    "name": "My Custom Model",
    "description": "Custom trained on my dataset",
    "type": "retinanet",
    "weights_path": "path/to/custom_model.pth",
    "framework": "pytorch"
}
```

## Summary

The model selection feature provides:
- ✅ Dynamic model discovery
- ✅ UI-based model switching
- ✅ Framework detection
- ✅ Real-time threshold updates
- ⏳ Extensible for new model types

Currently supports PyTorch RetinaNet, with infrastructure in place for Keras and TensorFlow models pending implementation.
