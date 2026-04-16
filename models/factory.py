from typing import Optional
from .model_provider import AbstractModel
from .retinanet_model import RetinaNetModel
from .owlv2_model import OWLv2Model


class ModelFactory:
    """
    Factory for creating object detection models.
    """

    @staticmethod
    def create_model(model_type: str, threshold: float = 0.5, weights_path: Optional[str] = None) -> AbstractModel:
        """
        Create a model instance based on type.

        Args:
            model_type: Type of model ('retinanet', 'owlv2', etc.)
            threshold: Detection threshold (default 0.5 for RetinaNet; recommended 0.1 for OWL-v2)
            weights_path: Optional path to custom weights or HuggingFace model name

        Returns:
            AbstractModel instance
        """
        if model_type.lower() == 'retinanet':
            model = RetinaNetModel(threshold=threshold)
            model.load_model(weights_path)
            return model
        elif model_type.lower() in ('owlv2', 'owl-v2', 'owl_v2'):
            model = OWLv2Model(threshold=threshold)
            model.load_model(weights_path)
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: retinanet, owlv2")
