"""
Módulo de detección para el sistema de reidentificación.
"""

from .face_detection import FaceDetection
from .body_detection import BodyDetection

__all__ = [
    'FaceDetection',
    'BodyDetection'
]
