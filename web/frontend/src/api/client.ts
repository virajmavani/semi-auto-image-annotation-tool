import axios from 'axios';
import { Detection, ImageInfo, BoundingBox, ModelInfo } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export const api = {
  async getModels(): Promise<{ models: ModelInfo[]; current_model: string; current_threshold: number }> {
    const response = await axios.get(`${API_BASE_URL}/api/models`);
    return response.data;
  },

  async getLabels(): Promise<string[]> {
    const response = await axios.get(`${API_BASE_URL}/api/labels`);
    return response.data.labels;
  },

  async uploadImage(file: File): Promise<ImageInfo> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/api/upload`, formData);
    return response.data;
  },

  async detectObjects(imagePath: string, selectedLabels: string[]): Promise<Detection[]> {
    const response = await axios.post(`${API_BASE_URL}/api/detect`, {
      image_path: imagePath,
      selected_labels: selectedLabels,
    });
    return response.data.detections;
  },

  async saveAnnotations(
    imageName: string,
    bboxes: BoundingBox[],
    imageWidth: number,
    imageHeight: number
  ): Promise<void> {
    await axios.post(`${API_BASE_URL}/api/save`, {
      image_name: imageName,
      bboxes: bboxes.map(bbox => ({
        x1: bbox.x1,
        y1: bbox.y1,
        x2: bbox.x2,
        y2: bbox.y2,
        label: bbox.label,
        score: bbox.score,
      })),
      image_width: imageWidth,
      image_height: imageHeight,
    });
  },

  async changeModel(modelId: string, threshold: number): Promise<void> {
    const formData = new FormData();
    formData.append('model_id', modelId);
    formData.append('threshold', threshold.toString());
    await axios.post(`${API_BASE_URL}/api/model/change`, formData);
  },

  async listImages(): Promise<ImageInfo[]> {
    const response = await axios.get(`${API_BASE_URL}/api/images`);
    return response.data.images;
  },

  getImageUrl(filename: string): string {
    return `${API_BASE_URL}/api/image/${filename}`;
  },

  async setDatasetDirectory(path: string): Promise<{ success: boolean; directory: string; image_count: number; images: ImageInfo[] }> {
    const formData = new FormData();
    formData.append('directory_path', path);
    const response = await axios.post(`${API_BASE_URL}/api/dataset/set`, formData);
    return response.data;
  },

  async listDatasetImages(): Promise<ImageInfo[]> {
    const response = await axios.get(`${API_BASE_URL}/api/dataset/images`);
    return response.data.images;
  },

  async loadAnnotations(filename: string): Promise<BoundingBox[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/annotations/${encodeURIComponent(filename)}`);
      return response.data.bboxes.map((bbox: any, idx: number) => ({
        id: `bbox-loaded-${Date.now()}-${idx}`,
        x1: bbox.x1,
        y1: bbox.y1,
        x2: bbox.x2,
        y2: bbox.y2,
        label: bbox.label,
        color: '#3b82f6', // Will be reassigned by the UI
      }));
    } catch (error) {
      return [];
    }
  },

  getDatasetImageUrl(filename: string): string {
    return `${API_BASE_URL}/api/dataset/image/${encodeURIComponent(filename)}`;
  },

  async browseDirectory(path: string): Promise<{ current_path: string; parent_path: string | null; entries: { name: string; path: string }[] }> {
    const response = await axios.get(`${API_BASE_URL}/api/browse`, { params: { path } });
    return response.data;
  },
};
