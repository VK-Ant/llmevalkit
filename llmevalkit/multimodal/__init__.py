"""llmevalkit multimodal evaluation module.

Metrics for OCR, audio transcription, image-text, vision QA,
document layout, and multimodal consistency.

Usage:
    from llmevalkit.multimodal import OCRAccuracy, AudioTranscriptionAccuracy
    from llmevalkit.multimodal import ImageTextAlignment, VisionQAAccuracy
    from llmevalkit.multimodal import DocumentLayoutAccuracy, MultimodalConsistency
"""

from llmevalkit.multimodal.metrics import (
    OCRAccuracy, AudioTranscriptionAccuracy,
    ImageTextAlignment, VisionQAAccuracy,
)
from llmevalkit.multimodal.additional_metrics import (
    DocumentLayoutAccuracy, MultimodalConsistency,
)

__all__ = [
    "OCRAccuracy", "AudioTranscriptionAccuracy",
    "ImageTextAlignment", "VisionQAAccuracy",
    "DocumentLayoutAccuracy", "MultimodalConsistency",
]
