from typing import List, Dict, Any, Optional
from .model_provider import AbstractModel


class OWLv2Model(AbstractModel):
    """OWL-v2 open-vocabulary object detection (google/owlv2-base-patch16-ensemble).

    Recommended threshold: 0.05–0.2 (much lower than RetinaNet's 0.5).
    Model weights (~600 MB) are downloaded from HuggingFace on first load and cached
    in ~/.cache/huggingface/.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.model = None
        self.processor = None

    def load_model(self, weights_path: str = None) -> None:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        name = weights_path or "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name)
        self.model.eval()

    def preprocess_image(self, image) -> Any:
        """Accept a torch uint8 CHW tensor or PIL Image; return PIL Image for use in predict()."""
        import torch
        import torchvision.transforms.functional as F
        if isinstance(image, torch.Tensor):
            return F.to_pil_image(image)
        return image  # already PIL

    def predict(self, preprocessed_image, text_queries: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not text_queries:
            return []

        import torch
        inputs = self.processor(
            text=[text_queries],
            images=preprocessed_image,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        w, h = preprocessed_image.size  # PIL: (width, height)
        target_sizes = torch.tensor([[h, w]])
        results = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

        detections = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.int().tolist()
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": text_queries[label_idx.item()],
                "score": score.item(),
            })
        return detections

    def get_labels(self) -> List[str]:
        return []  # no fixed label set

    def is_zero_shot(self) -> bool:
        return True
