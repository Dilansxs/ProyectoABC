"""
Clase base abstracta para preprocesadores de imágenes.

Define la interfaz común que todos los preprocesadores deben implementar.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    """
    Clase base abstracta para preprocesadores de imágenes.
    
    Todos los preprocesadores (BLP, HSH, LBP) deben heredar de esta clase
    e implementar los métodos abstractos.
    
    Attributes:
        name (str): Nombre del preprocesador.
        output_dim (int): Dimensionalidad del vector de salida.
    """
    
    def __init__(self, name: str, output_dim: int):
        """
        Inicializa el preprocesador base.
        
        Args:
            name (str): Nombre identificador del preprocesador.
            output_dim (int): Dimensionalidad del vector de características de salida.
        """
        self.name = name
        self.output_dim = output_dim
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen y la convierte en un vector de características.
        
        Args:
            image (np.ndarray): Imagen de entrada en formato numpy array (H, W, C) o (H, W).
        
        Returns:
            np.ndarray: Vector de características de dimensión (output_dim,).
        
        Raises:
            NotImplementedError: Debe ser implementado por las subclases.
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Preprocesa un lote de imágenes.
        
        Args:
            images (list): Lista de imágenes en formato numpy array.
        
        Returns:
            np.ndarray: Matriz de características de dimensión (N, output_dim).
        
        Raises:
            NotImplementedError: Debe ser implementado por las subclases.
        """
        pass
    
    def get_info(self) -> dict:
        """
        Obtiene información del preprocesador.
        
        Returns:
            dict: Diccionario con información del preprocesador.
        """
        return {
            'name': self.name,
            'output_dim': self.output_dim,
            'type': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', output_dim={self.output_dim})"
