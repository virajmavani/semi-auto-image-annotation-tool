import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.io.image import read_image
from PIL import Image
import numpy as np
from typing import List, Dict, Any
from .model_provider import AbstractModel


class RetinaNetModel(AbstractModel):
    """
    RetinaNet model implementation using torchvision.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.weights = None
        self.preprocess = None
        self.labels = None

    def load_model(self, weights_path: str = None) -> None:
        """
        Load RetinaNet model with default weights or from path.
        """
        if weights_path:
            # Load from custom path - assuming it's a state_dict
            self.weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT  # For meta
            self.model = retinanet_resnet50_fpn_v2(weights=None, box_score_thresh=self.threshold)
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        else:
            self.weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=self.weights, box_score_thresh=self.threshold)

        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL Image to tensor.
        """
        # Apply preprocessing
        batch = [self.preprocess(image)]
        return batch[0]  # Return single tensor, not batch

    def predict(self, preprocessed_image: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Run inference and return standardized detections.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Make batch
        batch = [preprocessed_image]

        # Run inference
        with torch.no_grad():
            prediction = self.model(batch)[0]

        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]

        # Filter by threshold (though model already filters, but to be safe)
        conf_mask = scores > self.threshold
        boxes = boxes[conf_mask]
        labels = labels[conf_mask]
        scores = scores[conf_mask]

        detections = []
        for box, label_idx, score in zip(boxes, labels, scores):
            label_name = self.labels[label_idx.item()]
            x1, y1, x2, y2 = box.int().tolist()
            detections.append({
                'box': [x1, y1, x2, y2],
                'label': label_name,
                'score': score.item()
            })

        return detections

    def get_labels(self) -> List[str]:
        """
        Return COCO class names.
        """
        if self.labels is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.labels