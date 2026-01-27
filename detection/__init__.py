"""
Módulo de detección para el sistema de reidentificación.
"""

from .face_detection import FaceDetection
from .body_detection import BodyDetection
from .view_classification import ViewClassification

__all__ = [
    'FaceDetection',
    'BodyDetection',
    'ViewClassification'
]
