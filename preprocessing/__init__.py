"""
Módulo de preprocesamiento y data augmentation para el sistema de reidentificación.

Este módulo proporciona funcionalidades para:
- Extracción de frames de videos a 10 FPS
- Detección y recorte de rostros y cuerpos
- Data augmentation (rotaciones, brillo, contraste, flip, ruido)
- Pipeline completo de preprocesamiento

Uso básico:
    from preprocessing import PreprocessingPipeline, create_pipeline
    
    # Crear pipeline
    pipeline = create_pipeline(
        dataset_path="data/dataset",
        output_path="data/datasetPros"
    )
    
    # Ejecutar pipeline completo
    stats = pipeline.run_full_pipeline(augmentation_multiplier=3)
"""

from .data_augmentation import DataAugmentation, AugmentationType
from .frame_extraction import FrameExtraction
from .preprocessing_pipeline import PreprocessingPipeline, create_pipeline
from .preprocessors import BasePreprocessor

__all__ = [
    # Data Augmentation
    'DataAugmentation',
    'AugmentationType',
    
    # Frame Extraction
    'FrameExtraction',
    
    # Pipeline
    'PreprocessingPipeline',
    'create_pipeline',
    
    # Base Preprocessor
    'BasePreprocessor'
]
