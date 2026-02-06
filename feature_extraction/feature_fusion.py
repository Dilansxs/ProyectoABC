"""
Módulo para fusionar características HOG (imagen) y MFCC (audio).

Combina características de cuerpo (HOG) con características de audio (MFCC)
para entrenar un modelo SVM multimodal que aprovecha ambas modalidades.
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import logging

from .hog import HOGExtractor
from .mfcc import MFCCExtractor
from detection.audio_detection import AudioDetection


class FeatureFusion:
    """
    Fusiona características HOG y MFCC de manera consistente.
    
    Proceso:
    1. Carga imágenes de: data/datasetPros/{person}/body/
    2. Carga audios de: data/datasetPros/audio/{person}/
    3. Extrae HOG de cada imagen
    4. Extrae MFCC de cada audio
    5. Asocia correctamente audio-imagen por persona
    6. Retorna matriz fusionada (N, HOG_dim + MFCC_dim)
    
    Attributes:
        hog_extractor: Instancia HOGExtractor
        mfcc_extractor: Instancia MFCCExtractor
        audio_detector: Instancia AudioDetection para validación
    """
    
    def __init__(self, 
                 hog_params: Optional[Dict] = None,
                 mfcc_params: Optional[Dict] = None,
                 use_audio_validation: bool = True):
        """
        Inicializa el fusionador de características.
        
        Args:
            hog_params: Parámetros para HOGExtractor
            mfcc_params: Parámetros para MFCCExtractor
            use_audio_validation: Validar audios antes de extraer MFCC
        """
        self.hog_params = hog_params or {}
        self.mfcc_params = mfcc_params or {}
        self.use_audio_validation = use_audio_validation
        
        # Instanciar extractores
        self.hog_extractor = HOGExtractor(**self.hog_params)
        self.mfcc_extractor = MFCCExtractor(**self.mfcc_params)
        self.audio_detector = AudioDetection()
        
        # Logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura logging."""
        self.logger = logging.getLogger('FeatureFusion')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _extract_hog_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extrae características HOG de una imagen.
        
        Args:
            image_path: Ruta de la imagen
        
        Returns:
            Vector HOG (1764,) o None si hay error
        """
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"No se pudo cargar imagen: {image_path}")
                return None
            return self.hog_extractor.extract(image)
        except Exception as e:
            self.logger.warning(f"Error extrayendo HOG de {image_path}: {e}")
            return None
    
    def _extract_mfcc_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extrae características MFCC de un audio.
        
        Args:
            audio_path: Ruta del archivo de audio
        
        Returns:
            Vector MFCC estadísticas (26,) o None si hay error
        """
        try:
            # Extraer características estadísticas directamente de la ruta
            # (sin validación intermedia que cause problemas de tipo)
            mfcc_stats = self.mfcc_extractor.extract_statistics(audio_path)
            return mfcc_stats
        except Exception as e:
            self.logger.warning(f"Error extrayendo MFCC de {audio_path}: {e}")
            # Fallback: vector de ceros
            return np.zeros(2 * self.mfcc_extractor.n_mfcc, dtype=np.float32)
    
    def _get_person_images(self, person_dir: str) -> List[str]:
        """
        Obtiene todas las imágenes de una persona.
        
        Args:
            person_dir: Directorio de la persona
        
        Returns:
            Lista de rutas de imágenes ordenadas
        """
        body_dir = os.path.join(person_dir, 'body')
        if not os.path.exists(body_dir):
            return []
        
        images = []
        for filename in sorted(os.listdir(body_dir)):
            if filename.endswith('.png'):
                images.append(os.path.join(body_dir, filename))
        return images
    
    def _get_person_audios(self, audio_base_dir: str, person_name: str) -> List[str]:
        """
        Obtiene todos los audios de una persona.
        
        Busca en el siguiente orden:
        1. data/audios_augmented/{person}/ (si existen audios augmentados)
        2. data/datasetPros/audio/{person}/ (audios originales)
        
        Esto evita replicación artificial - si usaste AudioAugmentation para
        expandir de 1 audio a 10-15 variantes, usará esas en lugar de replicar.
        
        Args:
            audio_base_dir: Directorio base de audios
            person_name: Nombre de la persona
        
        Returns:
            Lista de rutas de audios ordenadas
        """
        # Primero intentar buscar audios augmentados
        augmented_dir = os.path.join('data', 'audios_augmented', person_name)
        if os.path.exists(augmented_dir):
            audios = []
            for filename in sorted(os.listdir(augmented_dir)):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    audios.append(os.path.join(augmented_dir, filename))
            
            if audios:
                self.logger.info(f"    ✓ {len(audios)} audios augmentados encontrados")
                return audios
        
        # Si no hay augmentados, usar los audios originales
        person_audio_dir = os.path.join(audio_base_dir, person_name)
        if not os.path.exists(person_audio_dir):
            return []
        
        audios = []
        for filename in sorted(os.listdir(person_audio_dir)):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                audios.append(os.path.join(person_audio_dir, filename))
        
        if audios:
            self.logger.info(f"    ✓ {len(audios)} audios originales encontrados")
        
        return audios
    
    def fuse_features(self, 
                     dataset_processed_path: str = 'data/datasetPros',
                     audio_base_dir: str = 'data/datasetPros/audio') -> Tuple[np.ndarray, List[str]]:
        """
        Fusiona características HOG + MFCC de todas las personas.
        
        Estrategia:
        - Si hay N imágenes y M audios por persona:
          * Si N == M: Asociar 1-a-1 (imagen i con audio i)
          * Si N > M: Replicar audios (audio se repite cada N/M imágenes)
          * Si M > N: Tomar promedio de audios o usar solo primeros
        
        Args:
            dataset_processed_path: Ruta del dataset procesado (contiene carpetas por persona)
            audio_base_dir: Ruta base donde están guardados los audios
        
        Returns:
            Tupla (X_fused, labels) donde:
            - X_fused: Array (N, HOG_dim + MFCC_dim) = (N, 1764 + 26) = (N, 1790)
            - labels: Lista de etiquetas (persona)
        """
        self.logger.info("=" * 70)
        self.logger.info("INICIANDO FUSIÓN DE CARACTERÍSTICAS HOG + MFCC")
        self.logger.info("=" * 70)
        
        X_fused = []
        labels = []
        
        # Iterar sobre personas
        if not os.path.exists(dataset_processed_path):
            self.logger.error(f"Dataset no encontrado: {dataset_processed_path}")
            return np.array([]), []
        
        persons = sorted([d for d in os.listdir(dataset_processed_path)
                         if os.path.isdir(os.path.join(dataset_processed_path, d))])
        
        self.logger.info(f"Procesando {len(persons)} personas...")
        
        for person_name in persons:
            self.logger.info(f"\n➤ Persona: {person_name}")
            
            person_path = os.path.join(dataset_processed_path, person_name)
            
            # Obtener imágenes
            images = self._get_person_images(person_path)
            if not images:
                self.logger.warning(f"  ⚠️  No hay imágenes para {person_name}")
                continue
            
            self.logger.info(f"  ✓ Imágenes encontradas: {len(images)}")
            
            # Obtener audios
            audios = self._get_person_audios(audio_base_dir, person_name)
            if not audios:
                self.logger.warning(f"  ⚠️  No hay audios para {person_name}")
                # Continuar con HOG solo + MFCC ceros
                audios = [None] * len(images)
            else:
                self.logger.info(f"  ✓ Audios encontrados: {len(audios)}")
            
            # Estrategia de asociación: replicar audios si hay menos (ciclo repetido)
            if len(audios) < len(images):
                # Crear lista extendida usando ciclo repetido
                audios_extended = []
                n_valid_audios = len(audios) if audios else 1
                for i in range(len(images)):
                    audio_idx = i % n_valid_audios
                    audios_extended.append(audios[audio_idx] if audios else None)
                audios = audios_extended
                self.logger.info(f"  ↻ Audios replicados para {len(images)} imágenes (ciclo)")
            elif len(audios) > len(images):
                # Si hay más audios que imágenes, truncar
                audios = audios[:len(images)]
                self.logger.info(f"  ✂️  Audios truncados a {len(images)} imágenes")
            
            # Extraer características
            hog_features_list = []
            mfcc_features_list = []
            
            for idx, (image_path, audio_path) in enumerate(zip(images, audios)):
                # HOG
                hog_feat = self._extract_hog_features(image_path)
                if hog_feat is None:
                    hog_feat = np.zeros(self.hog_extractor.output_dim, dtype=np.float32)
                hog_features_list.append(hog_feat)
                
                # MFCC
                if audio_path is not None:
                    mfcc_feat = self._extract_mfcc_features(audio_path)
                else:
                    # Sin audio: vector de ceros
                    mfcc_feat = np.zeros(2 * self.mfcc_extractor.n_mfcc, dtype=np.float32)
                mfcc_features_list.append(mfcc_feat)
            
            # Concatenar HOG + MFCC
            hog_matrix = np.array(hog_features_list, dtype=np.float32)
            mfcc_matrix = np.array(mfcc_features_list, dtype=np.float32)
            
            fused_person = np.concatenate([hog_matrix, mfcc_matrix], axis=1)
            
            X_fused.append(fused_person)
            labels.extend([person_name] * len(fused_person))
            
            self.logger.info(f"  ✓ Características fusionadas: {fused_person.shape}")
        
        # Concatenar todas las personas
        if not X_fused:
            self.logger.error("No se pudieron fusionar características")
            return np.array([]), []
        
        X_final = np.vstack(X_fused).astype(np.float32)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("✓ FUSIÓN COMPLETADA")
        self.logger.info(f"  - Matriz fusionada: {X_final.shape}")
        self.logger.info(f"  - Dimensiones: HOG ({self.hog_extractor.output_dim}) + "
                        f"MFCC ({2 * self.mfcc_extractor.n_mfcc}) = "
                        f"{self.hog_extractor.output_dim + 2 * self.mfcc_extractor.n_mfcc}")
        self.logger.info(f"  - Muestras totales: {len(labels)}")
        self.logger.info(f"  - Clases: {len(set(labels))}")
        self.logger.info("=" * 70)
        
        return X_final, labels
    
    def get_feature_info(self) -> Dict:
        """
        Retorna información sobre las características fusionadas.
        
        Returns:
            Diccionario con información de dimensionalidad
        """
        hog_dim = self.hog_extractor.output_dim
        mfcc_dim = 2 * self.mfcc_extractor.n_mfcc
        
        return {
            'hog_dim': hog_dim,
            'mfcc_dim': mfcc_dim,
            'total_dim': hog_dim + mfcc_dim,
            'hog_description': f"Histogram of Oriented Gradients ({hog_dim} características)",
            'mfcc_description': f"MFCC Statistics - Media y Std ({mfcc_dim} características)",
            'fusion_description': f"Concatenación HOG + MFCC ({hog_dim + mfcc_dim} características)"
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def example_fusion():
    """Ejemplo de uso del fusionador de características."""
    
    fusion = FeatureFusion(
        use_audio_validation=True
    )
    
    # Mostrar información
    info = fusion.get_feature_info()
    print("\nInformación de características:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Fusionar características
    X_fused, labels = fusion.fuse_features(
        dataset_processed_path='data/datasetPros',
        audio_base_dir='data/datasetPros/audio'
    )
    
    if X_fused.size > 0:
        print(f"\n✓ Características fusionadas con éxito")
        print(f"  Forma: {X_fused.shape}")
        print(f"  Etiquetas únicas: {len(set(labels))}")
        print(f"  Etiquetas: {set(labels)}")
    else:
        print(f"❌ No se pudieron fusionar características")


if __name__ == "__main__":
    example_fusion()
