"""
Módulo para data augmentation de imágenes faciales y corporales.

Implementa técnicas de aumento de datos como rotaciones, ajustes de brillo,
y reflexiones horizontales para mejorar la robustez del modelo.
"""

import cv2
import numpy as np
import os
import random
import logging
from enum import Enum
from typing import Optional, List, Dict, Tuple
from pathlib import Path


class AugmentationType(Enum):
    """Enumeración de tipos de aumentos disponibles."""
    ROTATION = "rotation"
    BRIGHTNESS = "brightness"
    FLIP = "flip"
    CONTRAST = "contrast"
    NOISE = "noise"
    COMBINED = "combined"


class DataAugmentation:
    """
    Clase para aplicar transformaciones de data augmentation a imágenes.
    
    Attributes:
        rotation_range (tuple): Rango de ángulos de rotación en grados.
        brightness_range (tuple): Rango de ajuste de brillo (factor multiplicativo).
        contrast_range (tuple): Rango de ajuste de contraste.
        enable_flip (bool): Habilita reflexión horizontal.
    """
    
    def __init__(self, rotation_range: Tuple[float, float] = (-15, 15),
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 noise_level: float = 0.02):
        """
        Inicializa el módulo de data augmentation.
        
        Args:
            rotation_range (tuple): Rango de rotación (min, max) en grados.
            brightness_range (tuple): Rango de brillo (min, max) como factor.
            contrast_range (tuple): Rango de contraste (min, max) como factor.
            noise_level (float): Nivel de ruido gaussiano (0-1).
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_level = noise_level
        
        # Validar rangos de parámetros
        self._validate_params()
        
        # Configurar logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        self.logger = logging.getLogger('DataAugmentation')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _validate_params(self):
        """Valida los parámetros de configuración."""
        if self.rotation_range[0] > self.rotation_range[1]:
            raise ValueError("rotation_range: min debe ser menor que max")
        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("brightness_range: min debe ser menor que max")
        if self.brightness_range[0] < 0:
            raise ValueError("brightness_range: valores deben ser positivos")
        if self.contrast_range[0] > self.contrast_range[1]:
            raise ValueError("contrast_range: min debe ser menor que max")
        if not 0 <= self.noise_level <= 1:
            raise ValueError("noise_level debe estar entre 0 y 1")
    
    def apply_rotation(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Aplica rotación a una imagen.
        
        Args:
            image: Imagen en formato numpy array.
            angle (float): Ángulo de rotación en grados. Si es None, usa rango aleatorio.
        
        Returns:
            Imagen rotada en formato numpy array.
        """
        if angle is None:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Aplicar rotación
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                  borderMode=cv2.BORDER_REFLECT_101)
        
        return rotated
    
    def apply_brightness(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Ajusta el brillo de una imagen.
        
        Args:
            image: Imagen en formato numpy array.
            factor (float): Factor multiplicativo de brillo. Si es None, usa rango aleatorio.
        
        Returns:
            Imagen con brillo ajustado.
        """
        if factor is None:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        # Convertir a float para evitar overflow
        adjusted = image.astype(np.float32) * factor
        
        # Clipear valores al rango válido
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def apply_contrast(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Ajusta el contraste de una imagen.
        
        Args:
            image: Imagen en formato numpy array.
            factor (float): Factor de contraste. Si es None, usa rango aleatorio.
        
        Returns:
            Imagen con contraste ajustado.
        """
        if factor is None:
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # Calcular valor medio
        mean = np.mean(image)
        
        # Aplicar contraste
        adjusted = image.astype(np.float32)
        adjusted = (adjusted - mean) * factor + mean
        
        # Clipear valores al rango válido
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def apply_flip(self, image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """
        Aplica reflexión a una imagen.
        
        Args:
            image: Imagen en formato numpy array.
            horizontal (bool): Si es True, reflexión horizontal. Si es False, vertical.
        
        Returns:
            Imagen reflejada.
        """
        if horizontal:
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
    
    def apply_noise(self, image: np.ndarray, level: Optional[float] = None) -> np.ndarray:
        """
        Añade ruido gaussiano a una imagen.
        
        Args:
            image: Imagen en formato numpy array.
            level (float): Nivel de ruido. Si es None, usa nivel configurado.
        
        Returns:
            Imagen con ruido añadido.
        """
        if level is None:
            level = self.noise_level
        
        # Generar ruido gaussiano
        noise = np.random.normal(0, level * 255, image.shape).astype(np.float32)
        
        # Añadir ruido
        noisy = image.astype(np.float32) + noise
        
        # Clipear valores al rango válido
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def augment_image(self, image: np.ndarray, 
                      augmentation_type: AugmentationType = AugmentationType.COMBINED) -> np.ndarray:
        """
        Aplica aumento a una imagen según el tipo especificado.
        
        Args:
            image: Imagen en formato numpy array.
            augmentation_type (AugmentationType): Tipo de aumento a aplicar.
        
        Returns:
            Imagen aumentada.
        """
        if augmentation_type == AugmentationType.ROTATION:
            return self.apply_rotation(image)
        elif augmentation_type == AugmentationType.BRIGHTNESS:
            return self.apply_brightness(image)
        elif augmentation_type == AugmentationType.CONTRAST:
            return self.apply_contrast(image)
        elif augmentation_type == AugmentationType.FLIP:
            return self.apply_flip(image)
        elif augmentation_type == AugmentationType.NOISE:
            return self.apply_noise(image)
        elif augmentation_type == AugmentationType.COMBINED:
            # Aplicar combinación aleatoria de transformaciones
            result = image.copy()
            
            # 50% probabilidad de cada transformación
            if random.random() > 0.5:
                result = self.apply_rotation(result)
            if random.random() > 0.5:
                result = self.apply_brightness(result)
            if random.random() > 0.5:
                result = self.apply_contrast(result)
            if random.random() > 0.5:
                result = self.apply_flip(result)
            if random.random() > 0.7:  # Menos frecuente para ruido
                result = self.apply_noise(result)
            
            return result
        
        return image
    
    def generate_augmentations(self, image: np.ndarray, 
                               num_augmentations: int = 3) -> List[np.ndarray]:
        """
        Genera múltiples versiones aumentadas de una imagen.
        
        Args:
            image: Imagen original.
            num_augmentations (int): Número de augmentaciones a generar.
        
        Returns:
            Lista de imágenes aumentadas.
        """
        augmented_images = []
        
        for _ in range(num_augmentations):
            aug_image = self.augment_image(image, AugmentationType.COMBINED)
            augmented_images.append(aug_image)
        
        return augmented_images
    
    def augment_directory(self, input_dir: str, output_dir: str,
                          num_augmentations: int = 3,
                          preserve_originals: bool = True) -> Dict:
        """
        Aplica augmentation a todas las imágenes de un directorio.
        
        Args:
            input_dir: Directorio de entrada con imágenes.
            output_dir: Directorio de salida.
            num_augmentations: Número de augmentaciones por imagen.
            preserve_originals: Si True, copia también las originales.
        
        Returns:
            Diccionario con estadísticas.
        """
        stats = {
            'images_processed': 0,
            'augmentations_created': 0,
            'errors': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extensiones de imagen soportadas
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        # Obtener lista de imágenes
        images = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
        
        for img_name in images:
            img_path = os.path.join(input_dir, img_name)
            
            try:
                # Leer imagen
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"No se pudo leer la imagen: {img_path}")
                
                # Obtener nombre base sin extensión
                base_name = os.path.splitext(img_name)[0]
                
                # Guardar original si se requiere
                if preserve_originals:
                    original_path = os.path.join(output_dir, f"{base_name}_orig.png")
                    cv2.imwrite(original_path, image)
                
                # Generar augmentaciones
                augmented = self.generate_augmentations(image, num_augmentations)
                
                for i, aug_img in enumerate(augmented):
                    aug_path = os.path.join(output_dir, f"{base_name}_aug{i:02d}.png")
                    cv2.imwrite(aug_path, aug_img)
                    stats['augmentations_created'] += 1
                
                stats['images_processed'] += 1
                
            except Exception as e:
                error_msg = f"Error procesando {img_name}: {str(e)}"
                self.logger.error(error_msg)
                stats['errors'].append(error_msg)
                continue
        
        self.logger.info(f"Augmentation completado: {stats['images_processed']} imágenes, "
                         f"{stats['augmentations_created']} augmentaciones creadas")
        
        return stats
    
    def augment_dataset(self, dataset_path: str, output_path: str,
                        num_augmentations: int = 3) -> Dict:
        """
        Aplica augmentation a todo el dataset procesado.
        
        Estructura esperada:
        dataset_path/
        ├── persona1/
        │   ├── face/
        │   ├── front/
        │   └── back/
        └── ...
        
        Args:
            dataset_path: Ruta del dataset procesado.
            output_path: Ruta de salida (puede ser la misma si se quiere añadir).
            num_augmentations: Número de augmentaciones por imagen.
        
        Returns:
            Diccionario con estadísticas globales.
        """
        global_stats = {
            'persons_processed': 0,
            'total_images': 0,
            'total_augmentations': 0,
            'errors': []
        }
        
        if not os.path.exists(dataset_path):
            self.logger.error(f"Dataset no encontrado: {dataset_path}")
            return global_stats
        
        # Iterar sobre personas
        persons = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]
        
        self.logger.info(f"Aplicando augmentation a {len(persons)} personas...")
        
        for person_name in persons:
            person_input = os.path.join(dataset_path, person_name)
            person_output = os.path.join(output_path, person_name)
            
            # Procesar cada tipo de carpeta
            for folder_type in ['face', 'front', 'back']:
                input_folder = os.path.join(person_input, folder_type)
                output_folder = os.path.join(person_output, folder_type)
                
                if not os.path.exists(input_folder):
                    continue
                
                stats = self.augment_directory(
                    input_folder, output_folder,
                    num_augmentations=num_augmentations,
                    preserve_originals=False  # Ya están en el dataset
                )
                
                global_stats['total_images'] += stats['images_processed']
                global_stats['total_augmentations'] += stats['augmentations_created']
                global_stats['errors'].extend(stats['errors'])
            
            global_stats['persons_processed'] += 1
        
        self.logger.info("=" * 50)
        self.logger.info("DATA AUGMENTATION COMPLETADO")
        self.logger.info(f"Personas procesadas: {global_stats['persons_processed']}")
        self.logger.info(f"Imágenes procesadas: {global_stats['total_images']}")
        self.logger.info(f"Augmentaciones creadas: {global_stats['total_augmentations']}")
        if global_stats['errors']:
            self.logger.warning(f"Errores: {len(global_stats['errors'])}")
        self.logger.info("=" * 50)
        
        return global_stats
