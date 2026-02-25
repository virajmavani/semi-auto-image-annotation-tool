# Anno-Mage Web Features

## Modern UI/UX

### Dark Mode Design
- Eye-friendly dark theme with slate colors
- High contrast for better visibility
- Modern gradient accents (blue to purple)

### Responsive Layout
- Three-panel layout: Controls | Canvas | Annotations
- Flexible canvas sizing based on image dimensions
- Scrollable sidebars for long lists

## Interactive Canvas

### Drawing Features
- **Click & Drag**: Create new bounding boxes
- **Corner Handles**: Red circular handles appear when box is selected
- **Resize**: Drag any corner to resize the bounding box
- **Visual Feedback**: Dashed green preview while drawing
- **Crosshairs**: Precision lines follow cursor (like the tkinter version)

### Box Management
- **Color Coding**: Each box gets a unique color from palette
- **Labels**: Display label name above each box
- **Confidence Scores**: Show detection confidence percentage
- **Selection**: Click to select, visual highlighting

## AI-Powered Detection

### Auto-Detection
- RetinaNet-based object detection
- Pre-trained on COCO dataset (80 classes)
- Adjustable confidence threshold (0.0 - 1.0)
- Real-time detection processing

### Supported Objects
Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Annotation Management

### Label System
- **Pre-loaded COCO labels**: 80 common object categories
- **Custom labels**: Add your own categories
- **Multi-select**: Choose multiple labels for detection
- **Current label**: Select one for manual drawing

### Bounding Box List
- **Real-time updates**: See all boxes in right panel
- **Coordinates display**: Shows (x1, y1) → (x2, y2)
- **Confidence scores**: For auto-detected objects
- **Quick delete**: Trash icon for each box
- **Color matching**: Box color matches canvas

## Data Export

### Multiple Formats
1. **CSV Format**
   - Comma-separated values
   - Format: `image_path,x1,y1,x2,y2,label`
   - Appends to single file for dataset

2. **Pascal VOC XML**
   - Standard computer vision format
   - Compatible with many training frameworks
   - One XML file per image
   - Includes image dimensions

### Save Locations
```
web/backend/annotations/
├── annotations.csv
└── annotations_voc/
    ├── image1.xml
    ├── image2.xml
    └── ...
```

## Image Management

### Dataset Mode
- Load an entire local directory of images at once
- Thumbnail grid in the right panel for quick navigation
- Previous / Next buttons and ← → keyboard navigation
- Auto-saves annotations when moving between images
- Annotation cache keeps unsaved changes in memory while browsing
- Optional **auto-detect on load**: runs detection automatically each time an image opens

### Directory Browser
- Folder-picker modal launched from a button next to the directory path input
- Navigate the server's filesystem with single-click selection and double-click to open
- Up-arrow button to go to parent directory
- Selecting a directory fills the path input; clicking Load then opens the dataset

### Single Image Upload
- File input supports JPG, JPEG, PNG
- Automatic dimension detection
- Stored in `web/backend/uploads/`

### Image Display
- Automatic scaling to fit canvas
- Maintains aspect ratio
- Shows original dimensions and filename

## Controls & Settings

### Detection Controls
- **Threshold Slider**: Adjust from 0.0 to 1.0; model reloads at new threshold automatically
- **Auto Detect Button**: Run ML inference on the current image
- **Auto-detect on load**: Checkbox to run detection every time an image opens (skips images that already have annotations)
- **Loading State**: Visual feedback during processing

### Annotation Controls
- **Save**: Export annotations to CSV and XML (also Ctrl+S)
- **Clear All**: Remove all bounding boxes
- **Delete Individual**: Trash icon per box in the annotations panel

### Label Controls
- **Select All / Clear**: Bulk-toggle all COCO labels for detection
- **Add Custom Label**: Add labels beyond the COCO set; they appear in the drawing dropdown immediately

### Theme
- **Dark / Light toggle**: Button in the header; applies to all panels and the canvas border

### Status Messages
- Success notifications (green)
- Error messages (red)
- Auto-dismiss after 3 seconds
- Top-right positioning

## Keyboard & Mouse

### Mouse Interactions
- **Left Click + Drag** (canvas background): Draw new box
- **Click Box**: Select box (shows corner and edge handles)
- **Drag Corner / Edge Handle**: Resize selected box
- **Drag Box**: Move selected box anywhere on the canvas

### Keyboard Shortcuts
- **← / →**: Navigate to previous / next image (dataset mode)
- **Ctrl+S**: Save annotations for current image

### Visual Indicators
- **Dashed green preview**: While drawing a new box
- **White handles**: 4 corners + 4 edge midpoints shown on the selected box
- **Color-coded boxes**: Each box gets a unique color from the palette

## Performance Features

### Optimized Rendering
- Canvas-based rendering with Konva.js
- Hardware-accelerated graphics
- Smooth drag operations
- Efficient re-rendering

### API Architecture
- RESTful design
- Async operations
- Error handling
- CORS support for local development

## Comparison with Tkinter Version

### Enhanced Features
✅ Modern web interface (no desktop app needed)
✅ Better visual design with gradients and animations
✅ Responsive layout
✅ TypeScript for type safety
✅ RESTful API architecture
✅ Better color palette
✅ Cleaner code organization
✅ Browser-based (cross-platform)

### Feature Parity
✅ Image upload and annotation
✅ Auto-detection with ML models
✅ Manual bounding box drawing
✅ Label management
✅ CSV export
✅ Pascal VOC XML export
✅ Threshold adjustment
✅ Multiple label selection
✅ Precision view (crosshairs)
✅ Bounding box editing

### Differences
- Web version has an additional directory browser modal for navigating the filesystem
- Web version has a light/dark theme toggle
- Web version does not yet have the precision zoom panel (150×150 magnified view)
- Web version has better UI/UX and real-time slider feedback

## Future Enhancements

Potential features for future versions:
- Zoom and pan controls on the canvas
- Precision zoom panel (magnified cursor view, like the desktop version)
- Undo/redo functionality
- Export to other formats (YOLO, TFRecord, COCO JSON)
- Custom model upload via the UI
- Collaborative annotation
- User authentication
- Annotation history / versioning
- Image filters and preprocessing
