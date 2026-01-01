from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from PIL import Image


class AbstractModel(ABC):
    """
    Abstract base class for object detection models.
    All models should implement these methods to provide a unified interface.
    """

    @abstractmethod
    def load_model(self, weights_path: str = None) -> None:
        """
        Load the model with optional weights path.
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> Any:
        """
        Preprocess a PIL Image for model inference.
        Returns preprocessed data in the format expected by the model.
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_image: Any) -> List[Dict[str, Any]]:
        """
        Run inference on preprocessed image.
        Returns a list of detections, each as a dict with keys:
        - 'box': [x1, y1, x2, y2] in original image coordinates
        - 'label': str, class name
        - 'score': float, confidence score
        """
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """
        Return the list of class names supported by the model.
        Should be MS COCO labels for compatibility.
        """
        pass