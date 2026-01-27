"""
Módulo para la extracción de fotogramas de videos.
"""

import cv2
import os
from pathlib import Path


class FrameExtraction:
    """
    Clase encargada de la extracción de fotogramas a partir de archivos de video.
    """
    
    def __init__(self, fps=10, output_path=None):
        """
        Inicializa el extractor de fotogramas.
        """
        self.target_fps = fps
        self.output_path = output_path
        self.stats = {
            'videos_processed': 0,
            'frames_extracted': 0,
            'errors': []
        }
    
    def extract_frames(self, video_path, person_name, view_type):
        """
        Extrae fotogramas de un video específico.
        """
        if not os.path.exists(video_path):
            self.stats['errors'].append(f"Video no encontrado: {video_path}")
            return []
        
        # Crear directorio de salida
        output_dir = os.path.join(self.output_path, person_name, view_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.stats['errors'].append(f"No se pudo abrir el video: {video_path}")
            return []
        
        # Obtener información del video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps == 0:
            video_fps = 30  # Valor por defecto
        
        # Calcular intervalo de frames
        frame_interval = max(1, int(video_fps / self.target_fps))
        
        extracted_paths = []
        frame_count = 0
        saved_count = 0
        
        print(f"  Procesando: {os.path.basename(video_path)}")
        print(f"    FPS del video: {video_fps:.2f}, Extrayendo cada {frame_interval} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Guardar frame según intervalo
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                extracted_paths.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        self.stats['videos_processed'] += 1
        self.stats['frames_extracted'] += saved_count
        
        print(f"    Extraídos: {saved_count} frames de {total_frames} totales")
        
        return extracted_paths
    
    def process_dataset(self, dataset_path):
        """
        Procesa todos los videos del dataset original.
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset no encontrado en: {dataset_path}")
        
        print(f"\n{'='*60}")
        print("EXTRACCIÓN DE FOTOGRAMAS")
        print(f"{'='*60}")
        
        # Reiniciar estadísticas
        self.stats = {
            'videos_processed': 0,
            'frames_extracted': 0,
            'errors': [],
            'persons': {}
        }
        
        # Iterar sobre personas
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            print(f"\nPersona: {person_name}")
            self.stats['persons'][person_name] = {
                'videos': 0,
                'frames': 0
            }
            
            # Iterar sobre vistas (front/back)
            for view_type in ['front', 'back']:
                view_path = os.path.join(person_path, view_type)
                
                if not os.path.exists(view_path):
                    print(f"  ⚠ No se encontró carpeta: {view_type}")
                    continue
                
                # Procesar todos los videos en la vista
                video_files = [f for f in os.listdir(view_path) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                
                if not video_files:
                    print(f"  ⚠ No hay videos en: {view_type}")
                    continue
                
                print(f"\n  Vista: {view_type} ({len(video_files)} videos)")
                
                for video_file in video_files:
                    video_path = os.path.join(view_path, video_file)
                    frames_before = self.stats['frames_extracted']
                    
                    self.extract_frames(video_path, person_name, view_type)
                    
                    frames_extracted = self.stats['frames_extracted'] - frames_before
                    self.stats['persons'][person_name]['videos'] += 1
                    self.stats['persons'][person_name]['frames'] += frames_extracted
        
        return self.get_extraction_stats()
    
    def get_extraction_stats(self):
        """
        Obtiene estadísticas de la extracción realizada.
        """
        return {
            'total_videos': self.stats['videos_processed'],
            'total_frames': self.stats['frames_extracted'],
            'total_persons': len(self.stats['persons']),
            'persons_detail': self.stats['persons'],
            'errors': self.stats['errors']
        }