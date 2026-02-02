"""
Módulo de extracción de características para el sistema de reidentificación.
"""

from .body_features import BodyFeatureExtractor
from .feature_vector import FeatureVector

__all__ = [
    'BodyFeatureExtractor',
    'FeatureVector'
]
