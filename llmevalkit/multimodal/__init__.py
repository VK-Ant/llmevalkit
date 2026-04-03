"""llmevalkit multimodal evaluation module.

Basic metrics for OCR, audio transcription, image-text, and vision QA.

Usage:
    from llmevalkit.multimodal import OCRAccuracy, AudioTranscriptionAccuracy
    from llmevalkit.multimodal import ImageTextAlignment, VisionQAAccuracy
"""

from llmevalkit.multimodal.metrics import (
    OCRAccuracy, AudioTranscriptionAccuracy,
    ImageTextAlignment, VisionQAAccuracy,
)

__all__ = [
    "OCRAccuracy", "AudioTranscriptionAccuracy",
    "ImageTextAlignment", "VisionQAAccuracy",
]
