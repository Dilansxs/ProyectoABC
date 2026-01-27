
import cv2
import os
import numpy as np
from ultralytics import YOLO


class FaceDetection:
    """
    Detector de rostros usando YOLOv8n.
    
    Attributes:
        model: Modelo YOLOv8n cargado.
        confidence_threshold (float): Umbral de confianza mínimo.
    """
    
    def __init__(self, model='yolov8n', confidence_threshold=0.5):
        """
        Inicializa el detector de rostros.
        
        Args:
            model (str): Versión del modelo YOLO ('yolov8n', 'yolov8s', etc.)
            confidence_threshold (float): Umbral de confianza (0-1).
        """
        self.confidence_threshold = confidence_threshold
        
        try:
            # Cargar modelo YOLOv8n
            # Nota: El modelo se descargará automáticamente la primera vez
            print(f"[FaceDetection] Cargando modelo {model}...")
            self.model = YOLO(f'{model}.pt')
            print(f"[FaceDetection] Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"[ERROR] No se pudo cargar YOLOv8: {e}")
            print("[INFO] Asegúrate de haber instalado: pip install ultralytics")
            self.model = None
    
    def detect(self, image):
        """
        Detecta rostros en una imagen.
        
        Args:
            image (np.ndarray): Imagen en formato numpy array.
        
        Returns:
            list: Lista de tuplas (x, y, ancho, alto) para cada rostro.
        """
        if self.model is None:
            print("[WARN] Modelo no cargado, retornando detección vacía")
            return []
        
        # Ejecutar detección con YOLOv8
        results = self.model(image, verbose=False)
        
        detections = []
        
        # Procesar resultados
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Obtener clase y confianza
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Clase 0 en COCO dataset = 'person'
                # Para rostros específicos, podríamos usar un modelo fine-tuned
                # Por ahora detectamos personas completas
                if cls == 0 and conf >= self.confidence_threshold:
                    # Obtener coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convertir a formato (x, y, ancho, alto)
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    # Para rostros, tomamos la parte superior del cuerpo
                    # Aproximadamente el 30% superior
                    face_h = int(h * 0.3)
                    
                    detections.append((x, y, w, face_h))
        
        return detections
    
    def detect_and_crop(self, image):
        """
        Detecta rostros y recorta las regiones detectadas.
        
        Args:
            image (np.ndarray): Imagen en formato numpy array.
        
        Returns:
            list: Lista de imágenes recortadas (rostros detectados).
        """
        detections = self.detect(image)
        
        cropped_faces = []
        
        for (x, y, w, h) in detections:
            # Asegurar que las coordenadas estén dentro de la imagen
            y1 = max(0, y)
            y2 = min(image.shape[0], y + h)
            x1 = max(0, x)
            x2 = min(image.shape[1], x + w)
            
            # Recortar rostro
            face = image[y1:y2, x1:x2]
            
            if face.size > 0:  # Verificar que el recorte no esté vacío
                cropped_faces.append(face)
        
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
        
        for i, img in enumerate(images):
            detections = self.detect(img)
            results[i] = {
                'detections': detections,
                'count': len(detections)
            }
        
        return results
    
    def save_detections(self, image, output_path):
        """
        Detecta rostros y guarda los recortes.
        
        Args:
            image (np.ndarray): Imagen en formato numpy array.
            output_path (str): Ruta donde guardar los rostros detectados.
        
        Returns:
            list: Lista de rutas de archivos guardados.
        """
        os.makedirs(output_path, exist_ok=True)
        
        cropped_faces = self.detect_and_crop(image)
        
        saved_paths = []
        
        for i, face in enumerate(cropped_faces):
            filename = f"face_{i:04d}.jpg"
            filepath = os.path.join(output_path, filename)
            
            cv2.imwrite(filepath, face)
            saved_paths.append(filepath)
        
        return saved_paths