"""
Módulo para la extracción de fotogramas de videos por persona.

Este módulo procesa los videos que estén directamente dentro de cada carpeta
`data/{person}/{video}.mp4`, extrae frames, detecta cuerpos y genera
el dataset procesado (carpeta `body/` por persona).

"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Importar detectores del módulo detection
from detection import BodyDetection


class FrameExtraction:
    """
    Clase encargada de la extracción de fotogramas a partir de archivos de video.
    
    Extrae frames a 10 FPS, detecta cuerpos, y los guarda
    en la estructura de salida por persona (`body/`).
    
    Attributes:
        fps (int): Fotogramas por segundo a extraer del video.
        output_path (str): Ruta donde se almacenarán los fotogramas extraídos.
        body_resolution (tuple): Resolución de salida para cuerpos (256x512).
    """
    
    def __init__(self, fps: int = 10, output_path: Optional[str] = None,
                 body_resolution: Tuple[int, int] = (256, 512),
                 confidence_threshold: float = 0.5):
        """
        Inicializa el extractor de fotogramas.
        
        Args:
            fps (int): Fotogramas por segundo a extraer (por defecto 10).
            output_path (str): Ruta de salida para los fotogramas.
            body_resolution (tuple): Resolución para cuerpos (ancho, alto).
            confidence_threshold (float): Umbral de confianza para detecciones.
        """
        self.fps = fps
        self.output_path = output_path
        self.body_resolution = body_resolution
        self.confidence_threshold = confidence_threshold
        
        # Inicializar detectores
        self.body_detector = BodyDetection(confidence_threshold=confidence_threshold)
        
        # Estadísticas de extracción
        self.stats = {
            'videos_processed': 0,
            'total_frames_extracted': 0,
            'bodies_detected': 0,
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
    
    def extract_frames(self, video_path: str, person_name: str) -> Dict:
        """
        Extrae fotogramas de un video específico.
        
        Args:
            video_path (str): Ruta del archivo de video.
            person_name (str): Nombre de la persona para organizar carpetas.
        
        Returns:
            dict: Diccionario con estadísticas y rutas de fotogramas extraídos.
        """
        result = {
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

        # Carpeta para cuerpos
        body_path = os.path.join(person_output_path, 'body')
        os.makedirs(body_path, exist_ok=True)
        
        # Contador para nomenclatura
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
                    self.stats['bodies_detected'] += 1
                
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
        Procesa todos los videos dentro de cada carpeta de persona.
        
        Args:
            dataset_path (str): Ruta del dataset con estructura dataset/{persona}/{video_files}
        
        Returns:
            dict: Diccionario con estadísticas de extracción.
        """
        # Reiniciar estadísticas
        self.stats = {
            'videos_processed': 0,
            'total_frames_extracted': 0,
            'bodies_detected': 0,
            'frames_skipped': 0,
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
        
        self.logger.info(f"Procesando dataset con {len(persons)} personas...")
        
        for person_name in persons:
            person_path = os.path.join(dataset_path, person_name)
            self.logger.info(f"Procesando persona: {person_name}")

            # Buscar archivos de video en la carpeta de la persona
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
            videos = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(video_extensions)]

            for video_file in videos:
                video_path = os.path.join(person_path, video_file)

                try:
                    self.extract_frames(video_path, person_name)
                except Exception as e:
                    error_msg = f"Error procesando {video_path}: {str(e)}"
                    self.logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    continue
        
        self.logger.info("=" * 60)
        self.logger.info("✓ EXTRACCIÓN COMPLETADA")
        self.logger.info(f"Videos procesados: {self.stats['videos_processed']}")
        self.logger.info(f"Frames extraídos: {self.stats['total_frames_extracted']}")
        self.logger.info(f"Cuerpos detectados: {self.stats['bodies_detected']}")
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
