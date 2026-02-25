export interface BoundingBox {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  score?: number;
  color: string;
}

export interface ImageInfo {
  filename: string;
  path: string;
  width: number;
  height: number;
}

export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  score: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  framework: string;
  is_current: boolean;
}
