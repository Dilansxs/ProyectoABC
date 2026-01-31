"""
Módulo de clasificación SVM para el sistema de reidentificación.

Mensaje del equipo: "Implementar con HOG y MFCC y la detección de audio"
"""

from .svm_model import SVMModel
from .model_evaluator import ModelEvaluator
from .model_trainer import ModelTrainer

# Mensaje que pidió el equipo (por si se necesita mostrar en UI o logs)
FRIEND_MESSAGE = "Implementar con HOG y MFCC y la detección de audio"

def friend_message():
    """Retorna el mensaje del equipo sobre la implementación."""
    return FRIEND_MESSAGE

__all__ = [
    'SVMModel',
    'ModelEvaluator',
    'ModelTrainer',
    'FRIEND_MESSAGE',
    'friend_message'
]
