"""
Módulo para data augmentation de imágenes.
"""

import cv2
import numpy as np
import os
from enum import Enum


class AugmentationType(Enum):
    """Enumeración de tipos de aumentos disponibles."""
    ROTATION = "rotation"
    BRIGHTNESS = "brightness"
    FLIP = "flip"
    COMBINED = "combined"


class DataAugmentation:
    """
    Clase para aplicar transformaciones de data augmentation a imágenes.
    """
    
    def __init__(self, rotation_range=(-15, 15), brightness_range=(0.8, 1.2)):
        """
        Inicializa el módulo de data augmentation.
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.rng = np.random.RandomState(42)
    
    def apply_rotation(self, image, angle=None):
        """
        Aplica rotación a una imagen.
        """
        if angle is None:
            angle = self.rng.uniform(self.rotation_range[0], self.rotation_range[1])
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def apply_brightness(self, image, factor=None):
        """
        Ajusta el brillo de una imagen.
        """
        if factor is None:
            factor = self.rng.uniform(self.brightness_range[0], self.brightness_range[1])
        
        adjusted = image.astype(np.float32) * factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def apply_flip(self, image, horizontal=True):
        """
        Aplica reflexión a una imagen.
        """
        if horizontal:
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
    
    def augment_image(self, image, augmentation_type=AugmentationType.COMBINED):
        """
        Aplica aumento a una imagen según el tipo especificado.
        """
        if augmentation_type == AugmentationType.ROTATION:
            return self.apply_rotation(image)
        elif augmentation_type == AugmentationType.BRIGHTNESS:
            return self.apply_brightness(image)
        elif augmentation_type == AugmentationType.FLIP:
            return self.apply_flip(image)
        elif augmentation_type == AugmentationType.COMBINED:
            img = image.copy()
            
            if self.rng.random() > 0.5:
                img = self.apply_rotation(img)
            
            if self.rng.random() > 0.5:
                img = self.apply_brightness(img)
            
            if self.rng.random() > 0.7:
                img = self.apply_flip(img)
            
            return img
        
        return image
    
    def augment_batch(self, images, augmentation_type=AugmentationType.COMBINED, 
                     multiplier=3):
        """
        Aplica aumento a un lote de imágenes.
        """
        augmented = []
        
        for img in images:
            augmented.append(img)
            
            for _ in range(multiplier - 1):
                aug_img = self.augment_image(img, augmentation_type)
                augmented.append(aug_img)
        
        return augmented
    
    def augment_dataset(self, dataset_path, output_path, multiplier=3):
        """
        Aplica aumento a todas las imágenes de un dataset.
        """
        stats = {
            'original_images': 0,
            'augmented_images': 0,
            'persons': {}
        }
        
        print(f"\n{'='*60}")
        print("DATA AUGMENTATION")
        print(f"{'='*60}")
        print(f"Multiplicador: {multiplier}x")
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            print(f"\nPersona: {person_name}")
            stats['persons'][person_name] = {
                'original': 0,
                'augmented': 0
            }
            
            for view_type in ['front', 'back']:
                view_path = os.path.join(person_path, view_type)
                
                if not os.path.exists(view_path):
                    continue
                
                output_view_path = os.path.join(output_path, person_name, view_type)
                os.makedirs(output_view_path, exist_ok=True)
                
                image_files = [f for f in os.listdir(view_path) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                if not image_files:
                    continue
                
                print(f"  Vista {view_type}: {len(image_files)} imágenes")
                
                for img_file in image_files:
                    img_path = os.path.join(view_path, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    base_name = os.path.splitext(img_file)[0]
                    cv2.imwrite(
                        os.path.join(output_view_path, f"{base_name}_orig.jpg"),
                        img
                    )
                    stats['original_images'] += 1
                    stats['persons'][person_name]['original'] += 1
                    
                    for i in range(multiplier - 1):
                        aug_img = self.augment_image(img)
                        aug_filename = f"{base_name}_aug{i+1}.jpg"
                        cv2.imwrite(
                            os.path.join(output_view_path, aug_filename),
                            aug_img
                        )
                        stats['augmented_images'] += 1
                        stats['persons'][person_name]['augmented'] += 1
        
        stats['total_images'] = stats['original_images'] + stats['augmented_images']
        
        print(f"\n{'='*60}")
        print(f"Total imágenes originales: {stats['original_images']}")
        print(f"Total imágenes aumentadas: {stats['augmented_images']}")
        print(f"Total final: {stats['total_images']}")
        print(f"{'='*60}\n")
        
        return stats