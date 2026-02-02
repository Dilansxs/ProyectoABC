"""
Módulo para la extracción de características corporales.

Extrae vectores numéricos representativos de cuerpos capturando silueta,
proporciones y patrones visuales.

Soporta dos métodos de extracción:
- 'hog': Histogram of Oriented Gradients
- 'mfcc': Mel-Frequency Cepstral Coefficients (audio)
"""

import numpy as np
import cv2
import os
from .hog import HOGExtractor
from .mfcc import MFCCExtractor


class BodyFeatureExtractor:
    """
    Clase encargada de extraer características discriminativas de cuerpos.
    
    Attributes:
        method (str): Método de extracción ('hog', 'hsv', 'lbp').
        extractor: Instancia del extractor específico.
    """
    
    AVAILABLE_METHODS = {
        'hog': HOGExtractor,
        'mfcc': MFCCExtractor
    }
    
    def __init__(self, method='hog', **kwargs):
        """
        Inicializa el extractor de características corporales.
        
        Args:
            method (str): Método de extracción ('hog', 'mfcc'). Default: 'hog'.
            **kwargs: Parámetros adicionales para el extractor específico.
                     Para 'mfcc': n_mfcc, n_fft, hop_length, sr
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Método '{method}' no válido. Opciones disponibles: {list(self.AVAILABLE_METHODS.keys())}"
            )
        
        self.method = method
        self.extractor = self.AVAILABLE_METHODS[method](**kwargs)
    
    def extract(self, image):
        """
        Extrae características de una imagen corporal.
        
        Args:
            image (numpy.ndarray): Imagen en formato numpy array (cuerpo).
        
        Returns:
            numpy.ndarray: Vector de características.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Se espera numpy.ndarray, se recibió {type(image)}")
        
        if image.size == 0:
            raise ValueError("La imagen está vacía")
        
        return self.extractor.extract(image)
    
    def extract_batch(self, images):
        """
        Extrae características de un lote de imágenes corporales.
        
        Args:
            images (list): Lista de imágenes de cuerpos (numpy arrays).
        
        Returns:
            numpy.ndarray: Matriz de características de dimensión (N, feature_dim).
        """
        if not isinstance(images, (list, np.ndarray)):
            raise TypeError(f"Se espera lista o numpy.ndarray, se recibió {type(images)}")
        
        if len(images) == 0:
            raise ValueError("La lista de imágenes está vacía")
        
        return self.extractor.extract_batch(images)
    
    def extract_from_directory(self, directory_path):
        """
        Extrae características de todas las imágenes en un directorio.
        
        Args:
            directory_path (str): Ruta del directorio con imágenes.
        
        Returns:
            tuple: (features, file_paths, labels) con matriz, rutas y etiquetas.
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"El directorio no existe: {directory_path}")
        
        # Extensiones de imagen válidas
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        # Listar archivos de imagen
        image_files = []
        for f in os.listdir(directory_path):
            if f.lower().endswith(valid_extensions):
                image_files.append(f)
        
        if not image_files:
            raise ValueError(f"No se encontraron imágenes en: {directory_path}")
        
        # Cargar imágenes y extraer características
        features_list = []
        file_paths = []
        labels = []
        
        for img_file in image_files:
            img_path = os.path.join(directory_path, img_file)
            
            try:
                # Cargar imagen
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARN] No se pudo cargar: {img_path}")
                    continue
                
                # Extraer características
                features = self.extract(image)
                
                features_list.append(features)
                file_paths.append(img_path)
                
                # Etiquetar según nombre de directorio padre
                parent_dir = os.path.basename(os.path.dirname(img_path))
                labels.append(parent_dir)
                
            except Exception as e:
                print(f"[ERROR] Procesando {img_path}: {e}")
                continue
        
        # Convertir a arrays numpy
        features_matrix = np.array(features_list) if features_list else np.array([])
        
        return features_matrix, file_paths, labels
    
    def extract_from_lists(self, images_list):
        """
        Extrae características de una lista de imágenes corporales.

        Args:
            images_list (list): Lista de imágenes de cuerpos.

        Returns:
            numpy.ndarray: Matriz de características (N x feature_dim) o array vacío.
        """
        if not images_list:
            return np.array([])
        return self.extract_batch(images_list)

    
    def extract_and_save(self, image_path, output_path):
        """
        Extrae características y las guarda en archivo.
        
        Args:
            image_path (str): Ruta de la imagen.
            output_path (str): Ruta donde guardar las características.
        
        Returns:
            bool: True si se guardó exitosamente.
        """
        try:
            # Cargar imagen
            if not os.path.exists(image_path):
                raise ValueError(f"Imagen no encontrada: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Extraer características
            features = self.extract(image)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Guardar características
            np.save(output_path, features)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Guardando características: {e}")
            return False