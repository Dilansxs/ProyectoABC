"""
Pipeline de preprocesamiento que orquesta la extracción de fotogramas y data augmentation.

Este módulo coordina el flujo completo de preprocesamiento desde videos originales
hasta un dataset procesado y aumentado.
"""

import os
import shutil
import logging
from typing import Optional, Dict
from pathlib import Path

from .frame_extraction import FrameExtraction
from .data_augmentation import DataAugmentation


class PreprocessingPipeline:
    """
    Clase que coordina el pipeline completo de preprocesamiento.
    
    Transforma el dataset original de videos en un dataset procesado
    con imágenes de rostros y cuerpos detectados y aumentados.
    
    Estructura de entrada (dataset_path):
    dataset/
    ├── persona1/
    │   ├── front/
    │   │   └── video1.mp4
    │   └── back/
    │       └── video2.mp4
    └── ...
    
    Estructura de salida (output_path):
    datasetPros/
    ├── persona1/
    │   ├── face/
    │   │   └── img0001.png
    │   ├── front/
    │   │   └── img0001.png
    │   └── back/
    │       └── img0001.png
    └── ...
    
    Attributes:
        frame_extractor (FrameExtraction): Instancia del extractor de fotogramas.
        data_augmenter (DataAugmentation): Instancia del módulo de data augmentation.
        dataset_path (str): Ruta del dataset original.
        output_path (str): Ruta de salida para el dataset procesado.
    """
    
    def __init__(self, dataset_path: str, output_path: str, fps: int = 10,
                 face_resolution: tuple = (256, 256),
                 body_resolution: tuple = (256, 512),
                 confidence_threshold: float = 0.5):
        """
        Inicializa el pipeline de preprocesamiento.
        
        Args:
            dataset_path (str): Ruta del dataset original con videos.
            output_path (str): Ruta del dataset procesado de salida.
            fps (int): Fotogramas por segundo a extraer.
            face_resolution (tuple): Resolución para rostros (ancho, alto).
            body_resolution (tuple): Resolución para cuerpos (ancho, alto).
            confidence_threshold (float): Umbral de confianza para detecciones.
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.fps = fps
        self.face_resolution = face_resolution
        self.body_resolution = body_resolution
        self.confidence_threshold = confidence_threshold
        
        # Inicializar extractores y aumentadores
        self.frame_extractor = FrameExtraction(
            fps=fps,
            output_path=output_path,
            face_resolution=face_resolution,
            body_resolution=body_resolution,
            confidence_threshold=confidence_threshold
        )
        
        self.data_augmenter = DataAugmentation()
        
        # Estado del pipeline
        self.pipeline_status = {
            'extraction_completed': False,
            'augmentation_completed': False,
            'extraction_stats': None,
            'augmentation_stats': None
        }
        
        # Configurar logging
        self._setup_logging()
        
        # Crear estructura de carpetas de salida
        self._create_output_structure()
    
    def _setup_logging(self):
        """Configura el sistema de logging."""
        self.logger = logging.getLogger('PreprocessingPipeline')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _create_output_structure(self):
        """Crea la estructura de carpetas de salida."""
        os.makedirs(self.output_path, exist_ok=True)
        self.logger.info(f"Directorio de salida: {self.output_path}")
    
    def run_full_pipeline(self, augmentation_multiplier: int = 3) -> Dict:
        """
        Ejecuta el pipeline completo de preprocesamiento.
        
        1. Extrae fotogramas de videos a 10 FPS
        2. Detecta rostros y cuerpos
        3. Guarda imágenes redimensionadas
        4. Aplica data augmentation
        
        Args:
            augmentation_multiplier (int): Número de augmentaciones por imagen.
        
        Returns:
            dict: Estadísticas del procesamiento completo.
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO PIPELINE COMPLETO DE PREPROCESAMIENTO")
        self.logger.info("=" * 60)
        
        # Paso 1: Extracción de frames con detección
        self.logger.info("\n[PASO 1/2] Extracción de frames y detección...")
        extraction_stats = self.extract_frames_only()
        
        if extraction_stats['total_frames_extracted'] == 0:
            self.logger.warning("No se extrajeron frames. Abortando pipeline.")
            return {
                'success': False,
                'extraction_stats': extraction_stats,
                'augmentation_stats': None,
                'message': 'No se detectaron rostros ni cuerpos en los videos.'
            }
        
        # Paso 2: Data augmentation
        self.logger.info("\n[PASO 2/2] Aplicando data augmentation...")
        augmentation_stats = self.augment_only(augmentation_multiplier)
        
        # Compilar resultados finales
        final_stats = {
            'success': True,
            'extraction_stats': extraction_stats,
            'augmentation_stats': augmentation_stats,
            'summary': {
                'videos_processed': extraction_stats['videos_processed'],
                'total_original_images': extraction_stats['total_frames_extracted'],
                'total_augmented_images': augmentation_stats['total_augmentations'],
                'faces_generated': extraction_stats['faces_detected'],
                'bodies_front_generated': extraction_stats['bodies_front_detected'],
                'bodies_back_generated': extraction_stats['bodies_back_detected'],
                'total_errors': len(extraction_stats['errors']) + len(augmentation_stats['errors'])
            }
        }
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        self.logger.info("=" * 60)
        self.logger.info(f"Videos procesados: {final_stats['summary']['videos_processed']}")
        self.logger.info(f"Imágenes originales: {final_stats['summary']['total_original_images']}")
        self.logger.info(f"Imágenes aumentadas: {final_stats['summary']['total_augmented_images']}")
        self.logger.info(f"Dataset de salida: {self.output_path}")
        self.logger.info("=" * 60)
        
        return final_stats
    
    def extract_frames_only(self) -> Dict:
        """
        Ejecuta solo la etapa de extracción de fotogramas.
        
        Extrae frames de videos, detecta rostros y cuerpos,
        y guarda las imágenes en la estructura de salida.
        
        Returns:
            dict: Estadísticas de extracción.
        """
        self.logger.info("Iniciando extracción de frames...")
        
        # Procesar dataset
        stats = self.frame_extractor.process_dataset(self.dataset_path)
        
        # Actualizar estado
        self.pipeline_status['extraction_completed'] = True
        self.pipeline_status['extraction_stats'] = stats
        
        return stats
    
    def augment_only(self, augmentation_multiplier: int = 3) -> Dict:
        """
        Ejecuta solo la etapa de data augmentation.
        
        Requiere que la extracción haya sido completada primero.
        
        Args:
            augmentation_multiplier (int): Número de augmentaciones por imagen.
        
        Returns:
            dict: Estadísticas de data augmentation.
        """
        self.logger.info(f"Iniciando data augmentation (x{augmentation_multiplier})...")
        
        # Aplicar augmentation al dataset procesado
        stats = self.data_augmenter.augment_dataset(
            dataset_path=self.output_path,
            output_path=self.output_path,
            num_augmentations=augmentation_multiplier
        )
        
        # Actualizar estado
        self.pipeline_status['augmentation_completed'] = True
        self.pipeline_status['augmentation_stats'] = stats
        
        return stats
    
    def get_pipeline_status(self) -> Dict:
        """
        Obtiene el estado actual del pipeline.
        
        Returns:
            dict: Información del estado (etapas completadas, progreso, etc.)
        """
        # Contar archivos en el output
        file_counts = {
            'total_images': 0,
            'faces': 0,
            'front': 0,
            'back': 0,
            'persons': 0
        }
        
        if os.path.exists(self.output_path):
            persons = [d for d in os.listdir(self.output_path) 
                       if os.path.isdir(os.path.join(self.output_path, d))]
            file_counts['persons'] = len(persons)
            
            for person in persons:
                person_path = os.path.join(self.output_path, person)
                
                for folder in ['face', 'front', 'back']:
                    folder_path = os.path.join(person_path, folder)
                    if os.path.exists(folder_path):
                        count = len([f for f in os.listdir(folder_path) 
                                     if f.endswith('.png')])
                        file_counts[folder] += count
                        file_counts['total_images'] += count
        
        return {
            'extraction_completed': self.pipeline_status['extraction_completed'],
            'augmentation_completed': self.pipeline_status['augmentation_completed'],
            'extraction_stats': self.pipeline_status['extraction_stats'],
            'augmentation_stats': self.pipeline_status['augmentation_stats'],
            'output_path': self.output_path,
            'file_counts': file_counts
        }
    
    def clean_output(self, force: bool = False) -> bool:
        """
        Limpia los archivos de salida (sobrescribir).
        
        Args:
            force (bool): Si es True, elimina sin confirmar.
        
        Returns:
            bool: True si se limpió exitosamente.
        """
        if not os.path.exists(self.output_path):
            self.logger.info("Directorio de salida no existe. Nada que limpiar.")
            return True
        
        try:
            # Eliminar contenido del directorio
            shutil.rmtree(self.output_path)
            os.makedirs(self.output_path, exist_ok=True)
            
            # Reiniciar estado
            self.pipeline_status = {
                'extraction_completed': False,
                'augmentation_completed': False,
                'extraction_stats': None,
                'augmentation_stats': None
            }
            
            self.logger.info(f"Directorio de salida limpiado: {self.output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error limpiando directorio: {str(e)}")
            return False
    
    def generate_report(self) -> str:
        """
        Genera un reporte en texto del estado del pipeline.
        
        Returns:
            str: Reporte formateado.
        """
        status = self.get_pipeline_status()
        
        report = []
        report.append("=" * 60)
        report.append("REPORTE DEL PIPELINE DE PREPROCESAMIENTO")
        report.append("=" * 60)
        report.append("")
        report.append(f"Dataset origen: {self.dataset_path}")
        report.append(f"Dataset destino: {self.output_path}")
        report.append("")
        report.append("CONFIGURACIÓN:")
        report.append(f"  - FPS de extracción: {self.fps}")
        report.append(f"  - Resolución rostros: {self.face_resolution}")
        report.append(f"  - Resolución cuerpos: {self.body_resolution}")
        report.append(f"  - Umbral de confianza: {self.confidence_threshold}")
        report.append("")
        report.append("ESTADO DEL PIPELINE:")
        report.append(f"  - Extracción completada: {'Sí' if status['extraction_completed'] else 'No'}")
        report.append(f"  - Augmentation completada: {'Sí' if status['augmentation_completed'] else 'No'}")
        report.append("")
        report.append("CONTEO DE ARCHIVOS:")
        report.append(f"  - Personas: {status['file_counts']['persons']}")
        report.append(f"  - Rostros: {status['file_counts']['faces']}")
        report.append(f"  - Cuerpos frontales: {status['file_counts']['front']}")
        report.append(f"  - Cuerpos traseros: {status['file_counts']['back']}")
        report.append(f"  - Total imágenes: {status['file_counts']['total_images']}")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def create_pipeline(dataset_path: str = "data/dataset",
                    output_path: str = "data/datasetPros",
                    fps: int = 10) -> PreprocessingPipeline:
    """
    Función de utilidad para crear un pipeline con configuración predeterminada.
    
    Args:
        dataset_path: Ruta del dataset original.
        output_path: Ruta de salida.
        fps: Frames por segundo a extraer.
    
    Returns:
        PreprocessingPipeline configurado.
    """
    return PreprocessingPipeline(
        dataset_path=dataset_path,
        output_path=output_path,
        fps=fps,
        face_resolution=(256, 256),
        body_resolution=(256, 512),
        confidence_threshold=0.5
    )
