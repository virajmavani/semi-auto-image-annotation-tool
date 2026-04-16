import React, { useState, useEffect, useCallback } from 'react';
import { Upload, Save, Play, Trash2, ChevronLeft, ChevronRight, Tag, Moon, Sun, FolderOpen } from 'lucide-react';
import { AnnotationCanvas } from './components/AnnotationCanvas';
import { DirectoryBrowser } from './components/DirectoryBrowser';
import { BoundingBox, ImageInfo, ModelInfo } from './types';
import { api } from './api/client';
import { getColorForIndex } from './utils/colors';

function App() {
  const [currentImage, setCurrentImage] = useState<ImageInfo | null>(null);
  const [bboxes, setBboxes] = useState<BoundingBox[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [selectedLabels, setSelectedLabels] = useState<Set<string>>(new Set());
  const [currentLabel, setCurrentLabel] = useState<string | null>(null);
  const [customLabel, setCustomLabel] = useState('');
  const [threshold, setThreshold] = useState(0.5);
  const [isDetecting, setIsDetecting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [currentModelId, setCurrentModelId] = useState<string>('retinanet');

  // Dataset mode state
  const [datasetPath, setDatasetPath] = useState<string>('');
  const [imageList, setImageList] = useState<ImageInfo[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState<number>(0);
  const [isDatasetMode, setIsDatasetMode] = useState<boolean>(false);
  const [annotationCache, setAnnotationCache] = useState<Map<string, BoundingBox[]>>(new Map());

  // Directory browser modal
  const [showDirBrowser, setShowDirBrowser] = useState(false);

  // Auto-detect setting
  const [autoDetectOnLoad, setAutoDetectOnLoad] = useState<boolean>(false);

  // Zero-shot query state
  const [zeroShotQueries, setZeroShotQueries] = useState<string[]>([]);
  const [zeroShotInput, setZeroShotInput] = useState('');

  // Theme settings
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  // Derived: is the current model zero-shot?
  const isZeroShot = availableModels.find(m => m.id === currentModelId)?.is_zero_shot ?? false;

  useEffect(() => {
    loadModelsAndLabels();
  }, []);

  const loadModelsAndLabels = async () => {
    try {
      const modelsData = await api.getModels();
      setAvailableModels(modelsData.models);
      setCurrentModelId(modelsData.current_model);
      setThreshold(modelsData.current_threshold);

      const cocoLabels = await api.getLabels();
      setLabels(cocoLabels);
    } catch (error) {
      showMessage('error', 'Failed to load models and labels');
    }
  };

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 3000);
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const imageInfo = await api.uploadImage(file);
      setCurrentImage(imageInfo);
      setBboxes([]);
      setIsDatasetMode(false);
      showMessage('success', 'Image uploaded successfully');

      // Auto-detect if enabled
      const hasLabels = isZeroShot ? zeroShotQueries.length > 0 : selectedLabels.size > 0;
      if (autoDetectOnLoad && hasLabels) {
        await runAutoDetection(imageInfo);
      }
    } catch (error) {
      showMessage('error', 'Failed to upload image');
    }
  };

  const handleDetect = async () => {
    if (!currentImage) {
      showMessage('error', 'Please upload an image');
      return;
    }

    const labelsToSend = isZeroShot ? zeroShotQueries : Array.from(selectedLabels);
    if (labelsToSend.length === 0) {
      showMessage('error', isZeroShot ? 'Please enter at least one label query' : 'Please select labels');
      return;
    }

    setIsDetecting(true);
    try {
      const detections = await api.detectObjects(
        currentImage.path,
        labelsToSend
      );

      const newBboxes: BoundingBox[] = detections.map((det, idx) => ({
        id: `bbox-${Date.now()}-${idx}`,
        x1: det.x1,
        y1: det.y1,
        x2: det.x2,
        y2: det.y2,
        label: det.label,
        score: det.score,
        color: getColorForIndex(bboxes.length + idx),
      }));

      setBboxes([...bboxes, ...newBboxes]);
      showMessage('success', `Detected ${detections.length} objects`);
    } catch (error) {
      showMessage('error', 'Detection failed');
    } finally {
      setIsDetecting(false);
    }
  };

  const handleBboxCreate = useCallback((bbox: Omit<BoundingBox, 'id' | 'color'>) => {
    const newBbox: BoundingBox = {
      ...bbox,
      id: `bbox-${Date.now()}`,
      color: getColorForIndex(bboxes.length),
    };
    setBboxes(prev => [...prev, newBbox]);
  }, [bboxes.length]);

  const handleBboxUpdate = useCallback((id: string, updates: Partial<BoundingBox>) => {
    setBboxes(prev => prev.map(bbox =>
      bbox.id === id ? { ...bbox, ...updates } : bbox
    ));
  }, []);

  const handleBboxDelete = useCallback((id: string) => {
    setBboxes(prev => prev.filter(bbox => bbox.id !== id));
  }, []);

  const handleSave = async () => {
    if (!currentImage || bboxes.length === 0) {
      showMessage('error', 'No annotations to save');
      return;
    }

    try {
      await api.saveAnnotations(
        currentImage.filename,
        bboxes,
        currentImage.width,
        currentImage.height
      );
      showMessage('success', 'Annotations saved successfully');
    } catch (error) {
      showMessage('error', 'Failed to save annotations');
    }
  };

  const handleLabelToggle = (label: string) => {
    const newSelected = new Set(selectedLabels);
    if (newSelected.has(label)) {
      newSelected.delete(label);
    } else {
      newSelected.add(label);
    }
    setSelectedLabels(newSelected);
  };

  const handleAddCustomLabel = () => {
    if (customLabel && !labels.includes(customLabel)) {
      setLabels([...labels, customLabel]);
      setCustomLabel('');
      showMessage('success', 'Label added');
    }
  };

  const handleSelectAllLabels = () => {
    const allLabels = new Set(labels);
    setSelectedLabels(allLabels);
    showMessage('success', `All ${labels.length} labels selected`);
  };

  const handleDeselectAllLabels = () => {
    setSelectedLabels(new Set());
    showMessage('success', 'All labels deselected');
  };

  const handleClearAll = () => {
    setBboxes([]);
    showMessage('success', 'All annotations cleared');
  };

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModelId = e.target.value;
    try {
      await api.changeModel(newModelId, threshold);
      setCurrentModelId(newModelId);
      showMessage('success', `Model changed to ${availableModels.find(m => m.id === newModelId)?.name}`);

      // Reload labels as different models may have different label sets
      const cocoLabels = await api.getLabels();
      setLabels(cocoLabels);
      setSelectedLabels(new Set());
      setZeroShotQueries([]);
      setZeroShotInput('');
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to change model');
    }
  };

  const handleThresholdChange = async (newThreshold: number) => {
    setThreshold(newThreshold);
    try {
      await api.changeModel(currentModelId, newThreshold);
    } catch (error) {
      showMessage('error', 'Failed to update threshold');
    }
  };

  // Dataset mode handlers
  const handleLoadDataset = async () => {
    if (!datasetPath.trim()) {
      showMessage('error', 'Please enter a dataset directory path');
      return;
    }

    try {
      const result = await api.setDatasetDirectory(datasetPath);
      setImageList(result.images);
      setIsDatasetMode(true);
      setCurrentImageIndex(0);

      if (result.images.length > 0) {
        await loadImageAndAnnotations(result.images[0], 0);
        showMessage('success', `Loaded ${result.image_count} images from dataset`);
      }
    } catch (error: any) {
      showMessage('error', error.response?.data?.detail || 'Failed to load dataset');
    }
  };

  const loadImageAndAnnotations = async (imageInfo: ImageInfo, index: number) => {
    setCurrentImage(imageInfo);
    setCurrentImageIndex(index);

    // Check cache first
    if (annotationCache.has(imageInfo.filename)) {
      const cachedBboxes = annotationCache.get(imageInfo.filename)!;
      // Reassign colors
      const bboxesWithColors = cachedBboxes.map((bbox, idx) => ({
        ...bbox,
        color: getColorForIndex(idx),
      }));
      setBboxes(bboxesWithColors);
    } else {
      // Load from backend
      const loadedBboxes = await api.loadAnnotations(imageInfo.filename);
      const bboxesWithColors = loadedBboxes.map((bbox, idx) => ({
        ...bbox,
        color: getColorForIndex(idx),
      }));
      setBboxes(bboxesWithColors);

      // Auto-detect if enabled and no existing annotations
      const hasLabels = isZeroShot ? zeroShotQueries.length > 0 : selectedLabels.size > 0;
      if (autoDetectOnLoad && hasLabels && loadedBboxes.length === 0) {
        await runAutoDetection(imageInfo);
      }
    }
  };

  const runAutoDetection = async (imageInfo: ImageInfo) => {
    const labelsToSend = isZeroShot ? zeroShotQueries : Array.from(selectedLabels);
    if (labelsToSend.length === 0) return;

    setIsDetecting(true);
    try {
      const detections = await api.detectObjects(
        imageInfo.path,
        labelsToSend
      );

      const newBboxes: BoundingBox[] = detections.map((det, idx) => ({
        id: `bbox-${Date.now()}-${idx}`,
        x1: det.x1,
        y1: det.y1,
        x2: det.x2,
        y2: det.y2,
        label: det.label,
        score: det.score,
        color: getColorForIndex(idx),
      }));

      setBboxes(newBboxes);
    } catch (error) {
      console.error('Auto-detection failed:', error);
    } finally {
      setIsDetecting(false);
    }
  };

  const saveToCache = () => {
    if (currentImage) {
      const newCache = new Map(annotationCache);
      newCache.set(currentImage.filename, bboxes);
      setAnnotationCache(newCache);
    }
  };

  const handleNextImage = async () => {
    if (currentImageIndex < imageList.length - 1) {
      // Save current annotations
      if (bboxes.length > 0) {
        await handleSave();
      }
      saveToCache();

      // Load next image
      await loadImageAndAnnotations(imageList[currentImageIndex + 1], currentImageIndex + 1);
    }
  };

  const handlePreviousImage = async () => {
    if (currentImageIndex > 0) {
      // Save current annotations
      if (bboxes.length > 0) {
        await handleSave();
      }
      saveToCache();

      // Load previous image
      await loadImageAndAnnotations(imageList[currentImageIndex - 1], currentImageIndex - 1);
    }
  };

  const handleJumpToImage = async (index: number) => {
    if (index >= 0 && index < imageList.length && index !== currentImageIndex) {
      // Save current annotations
      if (bboxes.length > 0) {
        await handleSave();
      }
      saveToCache();

      // Load selected image
      await loadImageAndAnnotations(imageList[index], index);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!isDatasetMode) return;

      if (e.key === 'ArrowRight' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        handleNextImage();
      }
      if (e.key === 'ArrowLeft' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        handlePreviousImage();
      }
      if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleSave();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isDatasetMode, currentImageIndex, imageList, bboxes]);

  const handleAddZeroShotQuery = () => {
    const q = zeroShotInput.trim();
    if (q && !zeroShotQueries.includes(q)) {
      setZeroShotQueries([...zeroShotQueries, q]);
    }
    setZeroShotInput('');
  };

  const handleRemoveZeroShotQuery = (q: string) => {
    setZeroShotQueries(zeroShotQueries.filter(x => x !== q));
  };

  const themeClasses = {
    dark: {
      bg: 'bg-slate-900',
      bgSecondary: 'bg-slate-800',
      border: 'border-slate-700',
      text: 'text-slate-100',
      textSecondary: 'text-slate-400',
      input: 'bg-slate-900',
      hover: 'hover:bg-slate-800',
    },
    light: {
      bg: 'bg-gray-50',
      bgSecondary: 'bg-white',
      border: 'border-gray-300',
      text: 'text-gray-900',
      textSecondary: 'text-gray-600',
      input: 'bg-white',
      hover: 'hover:bg-gray-100',
    },
  };

  const currentTheme = themeClasses[theme];

  return (
    <div className={`min-h-screen ${currentTheme.bg} ${currentTheme.text}`}>
      {showDirBrowser && (
        <DirectoryBrowser
          initialPath={datasetPath || '~'}
          theme={theme}
          onSelect={(path) => {
            setDatasetPath(path);
            setShowDirBrowser(false);
          }}
          onClose={() => setShowDirBrowser(false)}
        />
      )}
      <header className={`${currentTheme.bgSecondary} border-b ${currentTheme.border} px-6 py-4`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-8">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Anno-Mage
            </h1>
            <p className={currentTheme.textSecondary}>Semi-Automatic Image Annotation Tool</p>
          </div>
          <div className="flex items-center gap-4">
            {/* Theme Toggle */}
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className={`p-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg ${currentTheme.hover} transition-colors`}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </header>

      {message && (
        <div className={`fixed top-20 right-6 px-6 py-3 rounded-lg shadow-lg z-50 ${
          message.type === 'success' ? 'bg-green-600' : 'bg-red-600'
        }`}>
          {message.text}
        </div>
      )}

      <div className="flex h-[calc(100vh-73px)]">
        <aside className={`w-80 ${currentTheme.bgSecondary} border-r ${currentTheme.border} p-4 overflow-y-auto`}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Dataset Directory</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  placeholder="/path/to/images"
                  className={`flex-1 px-3 py-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg focus:outline-none focus:border-blue-500 text-sm`}
                />
                <button
                  onClick={() => setShowDirBrowser(true)}
                  className={`px-3 py-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg hover:border-blue-500 transition-colors`}
                  title="Browse for directory"
                >
                  <FolderOpen size={16} />
                </button>
                <button
                  onClick={handleLoadDataset}
                  className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-sm text-white"
                >
                  Load
                </button>
              </div>
              {imageList.length > 0 && (
                <p className={`text-xs ${currentTheme.textSecondary} mt-1`}>
                  {imageList.length} images loaded
                </p>
              )}
            </div>

            <div className={`border-t ${currentTheme.border} pt-4`}>
              <label className="block text-sm font-medium mb-2">Or Upload Single Image</label>
              <label className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg cursor-pointer transition-colors text-white">
                <Upload size={18} />
                <span>Choose File</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Detection Model</label>
              <select
                value={currentModelId}
                onChange={handleModelChange}
                className={`w-full px-3 py-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg focus:outline-none focus:border-blue-500 text-sm`}
              >
                {availableModels.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.framework})
                  </option>
                ))}
              </select>
              <p className={`text-xs ${currentTheme.textSecondary} mt-1`}>
                {availableModels.find(m => m.id === currentModelId)?.description}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Detection Threshold</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={threshold}
                onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
                className="w-full"
              />
              <span className={`text-sm ${currentTheme.textSecondary}`}>{threshold.toFixed(2)}</span>
            </div>

            <div className="space-y-2">
              <button
                onClick={handleDetect}
                disabled={!currentImage || (isZeroShot ? zeroShotQueries.length === 0 : selectedLabels.size === 0) || isDetecting}
                className={`w-full flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-white ${!currentImage || (isZeroShot ? zeroShotQueries.length === 0 : selectedLabels.size === 0) || isDetecting ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <Play size={18} />
                <span>{isDetecting ? 'Detecting...' : 'Auto Detect'}</span>
              </button>

              <label className={`flex items-center gap-2 p-2 ${currentTheme.input} rounded-lg cursor-pointer ${currentTheme.hover} transition-colors`}>
                <input
                  type="checkbox"
                  checked={autoDetectOnLoad}
                  onChange={(e) => setAutoDetectOnLoad(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm">Auto-detect on image load</span>
              </label>
              {autoDetectOnLoad && (isZeroShot ? zeroShotQueries.length === 0 : selectedLabels.size === 0) && (
                <p className="text-xs text-yellow-500">
                  {isZeroShot ? 'Add label queries to enable auto-detection' : 'Select labels to enable auto-detection'}
                </p>
              )}
            </div>

            {isZeroShot ? (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium">Label Queries</label>
                  {zeroShotQueries.length > 0 && (
                    <button
                      onClick={() => setZeroShotQueries([])}
                      className={`px-2 py-1 text-xs ${currentTheme.input} border ${currentTheme.border} ${currentTheme.hover} rounded transition-colors`}
                    >
                      Clear All
                    </button>
                  )}
                </div>
                <div className="flex gap-2 mb-2">
                  <input
                    type="text"
                    value={zeroShotInput}
                    onChange={(e) => setZeroShotInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddZeroShotQuery()}
                    placeholder='e.g. "vintage ceramic teapot"'
                    className={`flex-1 px-3 py-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg focus:outline-none focus:border-blue-500 text-sm`}
                  />
                  <button
                    onClick={handleAddZeroShotQuery}
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-white text-sm"
                  >
                    Add
                  </button>
                </div>
                {zeroShotQueries.length > 0 && (
                  <div className={`flex flex-wrap gap-1 p-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg`}>
                    {zeroShotQueries.map(q => (
                      <span
                        key={q}
                        className="inline-flex items-center gap-1 px-2 py-1 bg-purple-600 text-white text-xs rounded-full"
                      >
                        {q}
                        <button
                          onClick={() => handleRemoveZeroShotQuery(q)}
                          className="ml-1 hover:text-red-200 leading-none"
                          aria-label={`Remove ${q}`}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}
                <p className={`text-xs ${currentTheme.textSecondary} mt-1`}>
                  Type any object description and press Enter or Add
                </p>
              </div>
            ) : (
              <>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium">Select Labels</label>
                    <div className="flex gap-1">
                      <button
                        onClick={handleSelectAllLabels}
                        className={`px-2 py-1 text-xs ${currentTheme.input} border ${currentTheme.border} ${currentTheme.hover} rounded transition-colors`}
                      >
                        Select All
                      </button>
                      <button
                        onClick={handleDeselectAllLabels}
                        className={`px-2 py-1 text-xs ${currentTheme.input} border ${currentTheme.border} ${currentTheme.hover} rounded transition-colors`}
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                  <div className={`space-y-1 max-h-60 overflow-y-auto ${currentTheme.input} rounded-lg p-2 border ${currentTheme.border}`}>
                    {labels.map(label => (
                      <label key={label} className={`flex items-center gap-2 p-1 ${currentTheme.hover} rounded cursor-pointer`}>
                        <input
                          type="checkbox"
                          checked={selectedLabels.has(label)}
                          onChange={() => handleLabelToggle(label)}
                          className="rounded"
                        />
                        <span className="text-sm">{label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Add Custom Label</label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={customLabel}
                      onChange={(e) => setCustomLabel(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAddCustomLabel()}
                      placeholder="Enter label"
                      className={`flex-1 px-3 py-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg focus:outline-none focus:border-blue-500`}
                    />
                    <button
                      onClick={handleAddCustomLabel}
                      className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors text-white"
                    >
                      <Tag size={18} />
                    </button>
                  </div>
                </div>
              </>
            )}

            <div>
              <label className="block text-sm font-medium mb-2">Current Label for Drawing</label>
              <select
                value={currentLabel || ''}
                onChange={(e) => setCurrentLabel(e.target.value || null)}
                className={`w-full px-3 py-2 ${currentTheme.input} border ${currentTheme.border} rounded-lg focus:outline-none focus:border-blue-500`}
              >
                <option value="">Select a label</option>
                {(isZeroShot ? zeroShotQueries : labels).map(label => (
                  <option key={label} value={label}>{label}</option>
                ))}
              </select>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleSave}
                disabled={!currentImage || bboxes.length === 0}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-white ${!currentImage || bboxes.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <Save size={18} />
                <span>Save</span>
              </button>
              <button
                onClick={handleClearAll}
                disabled={bboxes.length === 0}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors text-white ${bboxes.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <Trash2 size={18} />
                <span>Clear</span>
              </button>
            </div>
          </div>
        </aside>

        <main className="flex-1 p-6 flex flex-col">
          {currentImage ? (
            <>
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <h2 className={`text-lg font-medium ${currentTheme.textSecondary}`}>{currentImage.filename}</h2>
                  {isDatasetMode && (
                    <span className={`text-sm ${currentTheme.textSecondary}`}>
                      Image {currentImageIndex + 1} of {imageList.length}
                    </span>
                  )}
                </div>
                <span className={`text-sm ${currentTheme.textSecondary}`}>
                  {currentImage.width} x {currentImage.height}
                </span>
              </div>

              {isDatasetMode && (
                <div className="mb-4 flex items-center justify-center gap-4">
                  <button
                    onClick={handlePreviousImage}
                    disabled={currentImageIndex === 0}
                    className={`flex items-center gap-2 px-4 py-2 ${currentTheme.input} border ${currentTheme.border} ${currentTheme.hover} rounded-lg transition-colors ${currentImageIndex === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <ChevronLeft size={18} />
                    <span>Previous</span>
                  </button>

                  <span className={`text-sm ${currentTheme.textSecondary}`}>
                    Use ← → arrow keys to navigate
                  </span>

                  <button
                    onClick={handleNextImage}
                    disabled={currentImageIndex === imageList.length - 1}
                    className={`flex items-center gap-2 px-4 py-2 ${currentTheme.input} border ${currentTheme.border} ${currentTheme.hover} rounded-lg transition-colors ${currentImageIndex === imageList.length - 1 ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <span>Next</span>
                    <ChevronRight size={18} />
                  </button>
                </div>
              )}

              <div className="flex-1 flex items-center justify-center">
                <AnnotationCanvas
                  imageUrl={isDatasetMode ? api.getDatasetImageUrl(currentImage.filename) : api.getImageUrl(currentImage.filename)}
                  bboxes={bboxes}
                  onBboxCreate={handleBboxCreate}
                  onBboxUpdate={handleBboxUpdate}
                  onBboxDelete={handleBboxDelete}
                  selectedLabel={currentLabel}
                  canvasWidth={Math.min(currentImage.width, 900)}
                  canvasHeight={Math.min(currentImage.height, 600)}
                  theme={theme}
                />
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className={`text-center ${currentTheme.textSecondary}`}>
                <Upload size={64} className="mx-auto mb-4 opacity-50" />
                <p className="text-xl">Load a dataset or upload an image to start annotating</p>
              </div>
            </div>
          )}
        </main>

        <aside className={`w-80 ${currentTheme.bgSecondary} border-l ${currentTheme.border} p-4 overflow-y-auto`}>
          {isDatasetMode && imageList.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-medium mb-3">Images ({imageList.length})</h3>
              <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
                {imageList.map((img, idx) => (
                  <div
                    key={img.filename}
                    onClick={() => handleJumpToImage(idx)}
                    className={`cursor-pointer rounded-lg overflow-hidden border-2 transition-colors ${
                      idx === currentImageIndex
                        ? 'border-blue-500'
                        : `${currentTheme.border} ${currentTheme.hover}`
                    }`}
                  >
                    <div className={`aspect-video ${currentTheme.input} flex items-center justify-center overflow-hidden`}>
                      <img
                        src={api.getDatasetImageUrl(img.filename)}
                        alt={img.filename}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className={`p-1 ${currentTheme.input}`}>
                      <p className="text-xs truncate" title={img.filename}>
                        {img.filename}
                      </p>
                      <div className="flex items-center justify-between mt-1">
                        <span className={`text-xs ${currentTheme.textSecondary}`}>
                          {idx + 1}/{imageList.length}
                        </span>
                        {annotationCache.has(img.filename) && (
                          <span className="text-xs text-green-500">●</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <h3 className="text-lg font-medium mb-3">Annotations ({bboxes.length})</h3>
          <div className="space-y-2">
            {bboxes.map((bbox) => (
              <div
                key={bbox.id}
                className={`p-3 ${currentTheme.input} rounded-lg border ${currentTheme.border} ${currentTheme.hover} transition-colors`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium" style={{ color: bbox.color }}>
                    {bbox.label}
                  </span>
                  <button
                    onClick={() => handleBboxDelete(bbox.id)}
                    className="text-red-500 hover:text-red-400"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
                <div className={`text-xs ${currentTheme.textSecondary}`}>
                  ({bbox.x1}, {bbox.y1}) → ({bbox.x2}, {bbox.y2})
                  {bbox.score && <span className="ml-2">{(bbox.score * 100).toFixed(0)}%</span>}
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
