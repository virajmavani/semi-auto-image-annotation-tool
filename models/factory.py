from typing import Optional
from .model_provider import AbstractModel
from .retinanet_model import RetinaNetModel


class ModelFactory:
    """
    Factory for creating object detection models.
    """

    @staticmethod
    def create_model(model_type: str, threshold: float = 0.5, weights_path: Optional[str] = None) -> AbstractModel:
        """
        Create a model instance based on type.

        Args:
            model_type: Type of model ('retinanet', etc.)
            threshold: Detection threshold
            weights_path: Optional path to custom weights

        Returns:
            AbstractModel instance
        """
        if model_type.lower() == 'retinanet':
            model = RetinaNetModel(threshold=threshold)
            model.load_model(weights_path)
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: retinanet")
