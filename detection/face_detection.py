"""
Módulo para la detección de rostros en imágenes.

Implementa funcionalidades de detección de rostros utilizando modelos
de inteligencia artificial entrenados.
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO


class FaceDetection:
    
    def __init__(self, model='yolo', confidence_threshold=0.5):
        """
        Inicializa el detector de rostros.
        
        Args:
            model (str): Nombre del modelo a utilizar.
            confidence_threshold (float): Umbral de confianza mínimo.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.detections_count = 0
        
        # Cargar modelo de detección
        # Usar YOLOv8n (nano) para detección rápida de rostros
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Validar parámetros de configuración
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold debe estar entre 0 y 1")
        
        # Inicializar variables de seguimiento
        print(f"[FaceDetection] Modelo YOLOv8n cargado con umbral {confidence_threshold}")
    
    def detect(self, image):
        """
        Detecta rostros en una imagen.
        
        Args:
            image: Imagen en formato numpy array.
        
        Returns:
            list: Lista de tuplas (x, y, ancho, alto) para cada rostro detectado.
        """
        if image is None or image.size == 0:
            return []
        
        # Aplicar modelo de detección YOLO
        results = self.yolo_model(image, verbose=False)
        
        detections = []
        
        # Filtrar detecciones por confianza
        # Clase 0 en COCO dataset es 'person', detectamos personas 
        # y tomamos la parte superior como rostro
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtener confianza
                conf = float(box.conf[0])
                
                # Obtener clase (0 = person en COCO)
                cls = int(box.cls[0])
                
                # Filtrar por confianza y clase persona
                if conf >= self.confidence_threshold and cls == 0:
                    # Obtener coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convertir a formato (x, y, ancho, alto)
                    # Para rostros, tomamos la parte superior del body (20% superior)
                    width = x2 - x1
                    height = (y2 - y1) * 0.2  # 20% superior para aproximar rostro
                    
                    x = int(x1)
                    y = int(y1)
                    w = int(width)
                    h = int(height)
                    
                    detections.append((x, y, w, h))
        
        # Retornar coordenadas de rostros detectados
        self.detections_count += len(detections)
        return detections
    
    def detect_and_crop(self, image):
        """
        Detecta rostros y recorta las regiones detectadas.
        
        Args:
            image: Imagen en formato numpy array.
        
        Returns:
            list: Lista de imágenes recortadas (rostros detectados).
        """
        # Detectar rostros
        detections = self.detect(image)
        
        cropped_faces = []
        
        # Extraer coordenadas
        for (x, y, w, h) in detections:
            # Agregar margen
            margin = int(0.1 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            # Recortar regiones
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size > 0:
                cropped_faces.append(face_crop)
        
        # Retornar lista de imágenes recortadas
        return cropped_faces
    
    def process_batch(self, images):
        """
        Procesa un lote de imágenes para detectar rostros.
        
        Args:
            images (list): Lista de imágenes.
        
        Returns:
            dict: Diccionario con resultados por imagen.
        """
        results = {}
        
        # Iterar sobre imágenes
        for idx, image in enumerate(images):
            try:
                # Detectar rostros en cada imagen
                detections = self.detect(image)
                
                # Organizar resultados por imagen
                results[f'image_{idx}'] = {
                    'detections': detections,
                    'num_faces': len(detections),
                    'success': True
                }
            except Exception as e:
                results[f'image_{idx}'] = {
                    'detections': [],
                    'num_faces': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Retornar diccionario de resultados
        return results
    
    def save_detections(self, image, output_path):
        """
        Detecta rostros y guarda los recortes.
        
        Args:
            image: Imagen en formato numpy array.
            output_path (str): Ruta donde guardar los rostros detectados.
        
        Returns:
            list: Lista de rutas de archivos guardados.
        """
        # Detectar rostros
        cropped_faces = self.detect_and_crop(image)
        
        # Recortar y preparar imágenes
        os.makedirs(output_path, exist_ok=True)
        
        saved_paths = []
        
        # Guardar archivos
        for idx, face in enumerate(cropped_faces):
            filename = f"face_{idx:04d}.jpg"
            filepath = os.path.join(output_path, filename)
            cv2.imwrite(filepath, face)
            saved_paths.append(filepath)
        
        # Retornar lista de rutas
        return saved_paths