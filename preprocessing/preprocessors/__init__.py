"""
Módulo de preprocesadores de imágenes.

Contiene la clase base para preprocesadores utilizados en el sistema.
Los extractores de características específicos (LBP, HSH, HOG) están
en el módulo feature_extraction.
"""

from .base_preprocessor import BasePreprocessor

__all__ = [
    'BasePreprocessor'
]
