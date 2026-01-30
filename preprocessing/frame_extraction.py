"""
Módulo para la extracción de fotogramas de videos - SOLO VIDEOS FRONT.

Este módulo implementa la funcionalidad de extraer fotogramas de videos FRONTALES
a una frecuencia aproximada de 10 fotogramas por segundo, detectando
rostros y cuerpos para generar el dataset procesado.

"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Importar detectores del módulo detection
from detection import FaceDetection, BodyDetection


class FrameExtraction:
    """
    Clase encargada de la extracción de fotogramas a partir de archivos de video.
    
    Extrae frames a 10 FPS, detecta rostros y cuerpos, y los guarda
    en la estructura requerida para el dataset procesado.
    
    Attributes:
        fps (int): Fotogramas por segundo a extraer del video.
        output_path (str): Ruta donde se almacenarán los fotogramas extraídos.
        face_resolution (tuple): Resolución de salida para rostros (256x256).
        body_resolution (tuple): Resolución de salida para cuerpos (256x512).
    """
    
    def __init__(self, fps: int = 10, output_path: Optional[str] = None,
                 face_resolution: Tuple[int, int] = (256, 256),
                 body_resolution: Tuple[int, int] = (256, 512),
                 confidence_threshold: float = 0.5):
        """
        Inicializa el extractor de fotogramas.
        
        Args:
            fps (int): Fotogramas por segundo a extraer (por defecto 10).
            output_path (str): Ruta de salida para los fotogramas.
            face_resolution (tuple): Resolución para rostros (ancho, alto).
            body_resolution (tuple): Resolución para cuerpos (ancho, alto).
            confidence_threshold (float): Umbral de confianza para detecciones.
        """
        self.fps = fps
        self.output_path = output_path
        self.face_resolution = face_resolution
        self.body_resolution = body_resolution
        self.confidence_threshold = confidence_threshold
        
        # Inicializar detectores
        self.face_detector = FaceDetection(confidence_threshold=confidence_threshold)
        self.body_detector = BodyDetection(confidence_threshold=confidence_threshold)
        
        # Estadísticas de extracción
        self.stats = {
            'videos_processed': 0,
            'total_frames_extracted': 0,
            'faces_detected': 0,
            'bodies_front_detected': 0,
            'bodies_back_detected': 0,
            'frames_skipped': 0,
            'errors': []
        }
        
        # Configurar logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        self.logger = logging.getLogger('FrameExtraction')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Redimensiona una imagen manteniendo el aspecto y rellenando con negro.
        
        Args:
            image: Imagen a redimensionar.
            target_size: Tamaño objetivo (ancho, alto).
        
        Returns:
            Imagen redimensionada.
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # Calcular escala manteniendo aspecto
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crear canvas negro del tamaño objetivo
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Centrar imagen en canvas
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def _extract_face_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae y recorta el rostro de un frame.
        
        Args:
            frame: Frame del video.
        
        Returns:
            Imagen del rostro recortada y redimensionada, o None si no se detecta.
        """
        # Detectar rostros
        cropped_faces = self.face_detector.detect_and_crop(frame)
        
        if not cropped_faces:
            return None
        
        # Tomar el rostro con mayor área (asumimos es el principal)
        best_face = max(cropped_faces, key=lambda f: f.shape[0] * f.shape[1])
        
        # Redimensionar a resolución objetivo
        face_resized = self._resize_image(best_face, self.face_resolution)
        
        return face_resized
    
    def _extract_body_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae y recorta el cuerpo de un frame.
        
        Args:
            frame: Frame del video.
        
        Returns:
            Imagen del cuerpo recortada y redimensionada, o None si no se detecta.
        """
        # Detectar cuerpos
        cropped_bodies = self.body_detector.detect_and_crop(frame)
        
        if not cropped_bodies:
            return None
        
        # Tomar el cuerpo con mayor área (asumimos es el principal)
        best_body = max(cropped_bodies, key=lambda b: b.shape[0] * b.shape[1])
        
        # Redimensionar a resolución objetivo
        body_resized = self._resize_image(best_body, self.body_resolution)
        
        return body_resized
    
    def extract_frames(self, video_path: str, person_name: str, view_type: str) -> Dict:
        """
        Extrae fotogramas de un video específico (SOLO si es FRONT).
        
        Args:
            video_path (str): Ruta del archivo de video.
            person_name (str): Nombre de la persona para organizar carpetas.
            view_type (str): Tipo de vista ('front' o 'back').
        
        Returns:
            dict: Diccionario con estadísticas y rutas de fotogramas extraídos.
        
        Raises:
            ValueError: Si view_type no es 'front'.
        """
        # ✓ SOLO procesar videos FRONT
        if view_type.lower() not in ['front', 'frontal']:
            print(f"❌ [FrameExtraction] RECHAZADO: {video_path}")
            print(f"   Solo se aceptan videos FRONT. Se recibió: {view_type}")
            return {
                'faces_saved': [],
                'bodies_saved': [],
                'frames_processed': 0,
                'frames_with_detection': 0,
                'success': False,
                'error': f"Solo se aceptan videos FRONT. Se recibió: {view_type}"
            }
        
        result = {
            'faces_saved': [],
            'bodies_saved': [],
            'frames_processed': 0,
            'frames_with_detection': 0,
            'success': True,
            'error': None
        }
        
        # Verificar que el archivo existe
        if not os.path.exists(video_path):
            result['success'] = False
            result['error'] = f"Video no encontrado: {video_path}"
            self.stats['errors'].append(result['error'])
            return result
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result['success'] = False
            result['error'] = f"No se pudo abrir el video: {video_path}"
            self.stats['errors'].append(result['error'])
            return result
        
        # Obtener FPS del video original
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # Valor por defecto
        
        # Calcular intervalo de frames a extraer
        frame_interval = max(1, int(video_fps / self.fps))
        
        # Crear carpetas de salida
        person_output_path = os.path.join(self.output_path, person_name)
        
        # Solo extraer rostros de videos frontales
        if view_type == 'front':
            face_path = os.path.join(person_output_path, 'face')
            os.makedirs(face_path, exist_ok=True)
        
        body_path = os.path.join(person_output_path, view_type)
        os.makedirs(body_path, exist_ok=True)
        
        # Contadores para nomenclatura
        face_counter = self._get_next_counter(os.path.join(person_output_path, 'face')) if view_type == 'front' else 0
        body_counter = self._get_next_counter(body_path)
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Solo procesar frames en el intervalo deseado
            if frame_number % frame_interval == 0:
                result['frames_processed'] += 1
                detection_found = False
                
                # Extraer cuerpo
                body_img = self._extract_body_from_frame(frame)
                if body_img is not None:
                    body_filename = f"img{body_counter:04d}.png"
                    body_filepath = os.path.join(body_path, body_filename)
                    cv2.imwrite(body_filepath, body_img)
                    result['bodies_saved'].append(body_filepath)
                    body_counter += 1
                    detection_found = True
                    
                    if view_type == 'front':
                        self.stats['bodies_front_detected'] += 1
                    else:
                        self.stats['bodies_back_detected'] += 1
                
                # Extraer rostro solo de videos frontales
                if view_type == 'front':
                    face_img = self._extract_face_from_frame(frame)
                    if face_img is not None:
                        face_filename = f"img{face_counter:04d}.png"
                        face_filepath = os.path.join(face_path, face_filename)
                        cv2.imwrite(face_filepath, face_img)
                        result['faces_saved'].append(face_filepath)
                        face_counter += 1
                        detection_found = True
                        self.stats['faces_detected'] += 1
                
                if detection_found:
                    result['frames_with_detection'] += 1
                else:
                    self.stats['frames_skipped'] += 1
            
            frame_number += 1
        
        cap.release()
        
        self.stats['videos_processed'] += 1
        self.stats['total_frames_extracted'] += result['frames_with_detection']
        
        self.logger.info(
            f"Video procesado: {os.path.basename(video_path)} - "
            f"Frames: {result['frames_processed']}, "
            f"Detecciones: {result['frames_with_detection']}"
        )
        
        return result
    
    def _get_next_counter(self, folder_path: str) -> int:
        """
        Obtiene el siguiente número de contador para archivos en una carpeta.
        
        Args:
            folder_path: Ruta de la carpeta.
        
        Returns:
            Siguiente número disponible.
        """
        if not os.path.exists(folder_path):
            return 0
        
        existing_files = [f for f in os.listdir(folder_path) if f.startswith('img') and f.endswith('.png')]
        if not existing_files:
            return 0
        
        # Extraer números de los archivos existentes
        numbers = []
        for f in existing_files:
            try:
                num = int(f[3:7])  # img0001.png -> 0001
                numbers.append(num)
            except ValueError:
                continue
        
        return max(numbers) + 1 if numbers else 0
    
    def process_dataset(self, dataset_path: str) -> Dict:
        """
        Procesa todos los videos FRONT del dataset original.
        
        IMPORTANTE: Solo procesa videos FRONT. Rechaza automáticamente BACK.
        
        Args:
            dataset_path (str): Ruta del dataset con estructura dataset/{persona}/{front|back}
        
        Returns:
            dict: Diccionario con estadísticas de extracción.
        """
        # Reiniciar estadísticas
        self.stats = {
            'videos_processed': 0,
            'total_frames_extracted': 0,
            'faces_detected': 0,
            'bodies_front_detected': 0,
            'bodies_back_detected': 0,
            'frames_skipped': 0,
            'videos_rejected_back': 0,  # ← Contador de videos BACK rechazados
            'errors': []
        }
        
        # Verificar que el dataset existe
        if not os.path.exists(dataset_path):
            self.logger.error(f"Dataset no encontrado: {dataset_path}")
            self.stats['errors'].append(f"Dataset no encontrado: {dataset_path}")
            return self.stats
        
        # Iterar sobre carpetas de personas
        persons = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]
        
        self.logger.info(f"Procesando dataset con {len(persons)} personas (SOLO FRONT)...")
        
        for person_name in persons:
            person_path = os.path.join(dataset_path, person_name)
            self.logger.info(f"Procesando persona: {person_name}")
            
            # Iterar sobre vistas
            for view_type in ['front', 'back']:
                view_path = os.path.join(person_path, view_type)
                
                if not os.path.exists(view_path):
                    continue
                
                # ✓ SOLO procesar FRONT, rechazar BACK
                if view_type.lower() == 'back':
                    self.stats['videos_rejected_back'] += 1
                    continue
                
                # Buscar archivos de video
                video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
                videos = [f for f in os.listdir(view_path) 
                          if f.lower().endswith(video_extensions)]
                
                for video_file in videos:
                    video_path = os.path.join(view_path, video_file)
                    
                    try:
                        self.extract_frames(video_path, person_name, view_type)
                    except Exception as e:
                        error_msg = f"Error procesando {video_path}: {str(e)}"
                        self.logger.error(error_msg)
                        self.stats['errors'].append(error_msg)
                        continue
        
        self.logger.info("=" * 60)
        self.logger.info("✓ EXTRACCIÓN COMPLETADA")
        self.logger.info(f"Videos procesados (FRONT): {self.stats['videos_processed']}")
        self.logger.info(f"Frames extraídos: {self.stats['total_frames_extracted']}")
        self.logger.info(f"Rostros detectados: {self.stats['faces_detected']}")
        self.logger.info(f"Cuerpos frontales: {self.stats['bodies_front_detected']}")
        self.logger.info(f"Frames sin detección: {self.stats['frames_skipped']}")
        if self.stats['errors']:
            self.logger.warning(f"Errores: {len(self.stats['errors'])}")
        self.logger.info("=" * 60)
        
        return self.stats
    
    def get_extraction_stats(self) -> Dict:
        """
        Obtiene estadísticas de la extracción realizada.
        
        Returns:
            dict: Diccionario con estadísticas (frames extraídos, videos procesados, etc.)
        """
        return self.stats.copy()
