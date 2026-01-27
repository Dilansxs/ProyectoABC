import cv2
import os
import numpy as np
from ultralytics import YOLO


class BodyDetection:
    """
    Detector de cuerpos completos usando YOLOv8n.
    
    Attributes:
        model: Modelo YOLOv8n cargado.
        confidence_threshold (float): Umbral de confianza mínimo.
    """
    
    def __init__(self, model='yolov8n', confidence_threshold=0.5):
        """
        Inicializa el detector de cuerpos.
        
        Args:
            model (str): Versión del modelo YOLO ('yolov8n', 'yolov8s', etc.)
            confidence_threshold (float): Umbral de confianza (0-1).
        """
        self.confidence_threshold = confidence_threshold
        
        try:
            # Cargar modelo YOLOv8n
            print(f"[BodyDetection] Cargando modelo {model}...")
            self.model = YOLO(f'{model}.pt')
            print(f"[BodyDetection] Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"[ERROR] No se pudo cargar YOLOv8: {e}")
            print("[INFO] Asegúrate de haber instalado: pip install ultralytics")
            self.model = None
    
    def detect(self, image):
        """
        Detecta cuerpos en una imagen.
        
        Args:
            image (np.ndarray): Imagen en formato numpy array.
        
        Returns:
            list: Lista de tuplas (x, y, ancho, alto) para cada cuerpo detectado.
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
                if cls == 0 and conf >= self.confidence_threshold:
                    # Obtener coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convertir a formato (x, y, ancho, alto)
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    detections.append((x, y, w, h))
        
        return detections
    
    def detect_and_crop(self, image):
        """
        Detecta cuerpos y recorta las regiones detectadas.
        
        Args:
            image (np.ndarray): Imagen en formato numpy array.
        
        Returns:
            list: Lista de imágenes recortadas (cuerpos detectados).
        """
        detections = self.detect(image)
        
        cropped_bodies = []
        
        for (x, y, w, h) in detections:
            # Asegurar que las coordenadas estén dentro de la imagen
            y1 = max(0, y)
            y2 = min(image.shape[0], y + h)
            x1 = max(0, x)
            x2 = min(image.shape[1], x + w)
            
            # Recortar cuerpo
            body = image[y1:y2, x1:x2]
            
            if body.size > 0:  # Verificar que el recorte no esté vacío
                cropped_bodies.append(body)
        
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
        
        for i, img in enumerate(images):
            detections = self.detect(img)
            results[i] = {
                'detections': detections,
                'count': len(detections)
            }
        
        return results
    
    def save_detections(self, image, output_path):
        """
        Detecta cuerpos y guarda los recortes.
        
        Args:
            image (np.ndarray): Imagen en formato numpy array.
            output_path (str): Ruta donde guardar los cuerpos detectados.
        
        Returns:
            list: Lista de rutas de archivos guardados.
        """
        os.makedirs(output_path, exist_ok=True)
        
        cropped_bodies = self.detect_and_crop(image)
        
        saved_paths = []
        
        for i, body in enumerate(cropped_bodies):
            filename = f"body_{i:04d}.jpg"
            filepath = os.path.join(output_path, filename)
            
            cv2.imwrite(filepath, body)
            saved_paths.append(filepath)
        
        return saved_paths
    
    def detect_from_video(self, video_path, output_path, fps=10):
        """
        Detecta y extrae cuerpos de un video.
        
        Args:
            video_path (str): Ruta del video.
            output_path (str): Ruta de salida para los cuerpos detectados.
            fps (int): Frames por segundo a procesar.
        
        Returns:
            dict: Estadísticas de detección.
        """
        os.makedirs(output_path, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'No se pudo abrir el video: {video_path}'}
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            video_fps = 30
        
        frame_interval = max(1, int(video_fps / fps))
        
        frame_count = 0
        saved_count = 0
        total_detections = 0
        
        print(f"  Procesando video: {os.path.basename(video_path)}")
        print(f"    FPS del video: {video_fps:.2f}")
        print(f"    Procesando cada {frame_interval} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Detectar cuerpos en el frame
                bodies = self.detect_and_crop(frame)
                
                # Guardar cada cuerpo detectado
                for i, body in enumerate(bodies):
                    filename = f"frame_{frame_count:06d}_body_{i:02d}.jpg"
                    filepath = os.path.join(output_path, filename)
                    cv2.imwrite(filepath, body)
                    saved_count += 1
                
                total_detections += len(bodies)
            
            frame_count += 1
        
        cap.release()
        
        return {
            'frames_processed': frame_count // frame_interval,
            'total_frames': frame_count,
            'bodies_detected': total_detections,
            'bodies_saved': saved_count
        }