"""
Módulo para la detección de cuerpos en imágenes.

Implementa funcionalidades de detección de cuerpos completos utilizando
modelos de detección de objetos basados en redes neuronales.
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO


class BodyDetection:
    
    def __init__(self, model='yolo', confidence_threshold=0.5):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.detections_count = 0
        
        # Cargar modelo de detección de cuerpos
        # Usar YOLOv8n (nano) para detección rápida y eficiente
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Validar parámetros de configuración
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold debe estar entre 0 y 1")
        
        # Inicializar variables de seguimiento
        print(f"[BodyDetection] ✓ Modelo YOLOv8n cargado con umbral {confidence_threshold}")
    
    def detect(self, image):
        """
        Detecta cuerpos en una imagen.
        
        Args:
            image: Imagen en formato numpy array.
        
        Returns:
            list: Lista de tuplas (x, y, ancho, alto) para cada cuerpo detectado.
        """
        if image is None or image.size == 0:
            return []
        
        # Aplicar modelo de detección
        results = self.yolo_model(image, verbose=False)
        
        detections = []
        
        # Filtrar detecciones por confianza
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtener confianza
                conf = float(box.conf[0])
                
                # Obtener clase (0 = person en COCO dataset)
                cls = int(box.cls[0])
                
                # Filtrar por confianza y clase persona
                if conf >= self.confidence_threshold and cls == 0:
                    # Obtener coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convertir a formato (x, y, ancho, alto)
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    detections.append((x, y, w, h))
        
        # Retornar coordenadas de cuerpos detectados
        self.detections_count += len(detections)
        return detections
    
    def detect_and_crop(self, image):
        """
        Detecta cuerpos y recorta las regiones detectadas.
        
        Args:
            image: Imagen en formato numpy array.
        
        Returns:
            list: Lista de imágenes recortadas (cuerpos detectados).
        """
        # Detectar cuerpos
        detections = self.detect(image)
        
        cropped_bodies = []
        
        # Extraer coordenadas
        for (x, y, w, h) in detections:
            # Agregar margen
            margin = int(0.05 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            # Recortar regiones
            body_crop = image[y1:y2, x1:x2]
            if body_crop.size > 0:
                cropped_bodies.append(body_crop)
        
        # Retornar lista de imágenes recortadas
        return cropped_bodies
    
    def process_batch(self, images):
        """
        Procesa un lote de imágenes para detectar cuerpos.
        
        Args:
            images (list): Lista de imágenes.
        
        Returns:
            dict: Diccionario con resultados por imagen.
        """
        results = {}
        
        # Iterar sobre imágenes
        for idx, image in enumerate(images):
            try:
                # Detectar cuerpos en cada imagen
                detections = self.detect(image)
                
                # Organizar resultados por imagen
                results[f'image_{idx}'] = {
                    'detections': detections,
                    'num_bodies': len(detections),
                    'success': True
                }
            except Exception as e:
                results[f'image_{idx}'] = {
                    'detections': [],
                    'num_bodies': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Retornar diccionario de resultados
        return results
    
    def save_detections(self, image, output_path):
        """
        Detecta cuerpos y guarda los recortes.
        
        Args:
            image: Imagen en formato numpy array.
            output_path (str): Ruta donde guardar los cuerpos detectados.
        
        Returns:
            list: Lista de rutas de archivos guardados.
        """
        # Detectar cuerpos
        cropped_bodies = self.detect_and_crop(image)
        
        # Recortar y preparar imágenes
        os.makedirs(output_path, exist_ok=True)
        
        saved_paths = []
        
        # Guardar archivos
        for idx, body in enumerate(cropped_bodies):
            filename = f"body_{idx:04d}.jpg"
            filepath = os.path.join(output_path, filename)
            cv2.imwrite(filepath, body)
            saved_paths.append(filepath)
        
        # Retornar lista de rutas
        return saved_paths