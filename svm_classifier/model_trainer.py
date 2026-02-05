from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .svm_model import SVMModel
from .model_evaluator import ModelEvaluator

import numpy as np
from typing import Optional, Dict, Any
import logging

# Extractors y detectores
from feature_extraction.hog import HOGExtractor
from feature_extraction.mfcc import MFCCExtractor
from feature_extraction.feature_fusion import FeatureFusion
from detection.audio_detection import AudioDetection


class ModelTrainer:
    """
    Entrenador para modelo SVM multimodal (HOG + MFCC).
    
    Soporta dos modos:
    1. FUSIÓN AUTOMÁTICA: Carga datos de data/datasetPros y data/datasetPros/audio
       Usa FeatureFusion para combinar HOG (imagen) + MFCC (audio)
    
    2. MANUAL: Acepta características ya calculadas o brutos imágenes/audios
    
    Características fusionadas: (1764 + 26) = 1790 dimensiones
    - HOG: 1764 (características de forma del cuerpo)
    - MFCC: 26 (media y desviación estándar de 13 coeficientes)
    """
    
    def __init__(
        self,
        train_test_split_ratio: float = 0.8,
        use_fusion: bool = True,
        hog_params: Optional[Dict[str, Any]] = None,
        mfcc_params: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa el entrenador.
        
        Args:
            train_test_split_ratio: Ratio train/test (default 0.8)
            use_fusion: Usar FeatureFusion automático (default True)
            hog_params: Parámetros para HOG
            mfcc_params: Parámetros para MFCC
        """
        self.train_test_split_ratio = train_test_split_ratio
        self.use_fusion = use_fusion
        self.hog_params = hog_params or {}
        self.mfcc_params = mfcc_params or {}
        
        # Instanciar fusionador si está habilitado
        if self.use_fusion:
            self.feature_fusion = FeatureFusion(
                hog_params=self.hog_params,
                mfcc_params=self.mfcc_params,
                use_audio_validation=True
            )
        else:
            self.feature_fusion = None
        
        # Logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura logging."""
        self.logger = logging.getLogger('ModelTrainer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def load_fused_features(self, 
                           dataset_processed_path: str = 'data/datasetPros',
                           audio_base_dir: str = 'data/datasetPros/audio') -> tuple:
        """
        Carga características fusionadas HOG + MFCC usando FeatureFusion.
        
        Args:
            dataset_processed_path: Ruta de data/datasetPros
            audio_base_dir: Ruta de data/datasetPros/audio
        
        Returns:
            Tupla (X, labels) con características fusionadas
        """
        if not self.use_fusion:
            raise ValueError("use_fusion debe ser True para usar load_fused_features")
        
        self.logger.info("Cargando características fusionadas...")
        X, labels = self.feature_fusion.fuse_features(
            dataset_processed_path=dataset_processed_path,
            audio_base_dir=audio_base_dir
        )
        
        return X, labels
    
    def train(self, X=None, y=None, validate=True,
              dataset_processed_path: str = 'data/datasetPros',
              audio_base_dir: str = 'data/datasetPros/audio'):
        """
        Entrena el modelo SVM.
        
        Dos modos:
        1. AUTOMÁTICO: Si X=None, carga características fusionadas del disco
        2. MANUAL: Si X y y se proporcionan, usa esas características
        
        Args:
            X: Características (None para cargar automáticamente)
            y: Etiquetas (None para cargar automáticamente)
            validate: Hacer split train/test
            dataset_processed_path: Ruta de data/datasetPros
            audio_base_dir: Ruta de data/datasetPros/audio
        
        Returns:
            Tupla (model, metrics)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("ENTRENAMIENTO DEL MODELO SVM MULTIMODAL (HOG + MFCC)")
        self.logger.info("=" * 70)
        
        # Cargar características si no se proporcionan
        if X is None or y is None:
            if not self.use_fusion:
                raise ValueError("Se deben proporcionar X e y, o habilitar use_fusion=True")
            X, y = self.load_fused_features(
                dataset_processed_path=dataset_processed_path,
                audio_base_dir=audio_base_dir
            )
        
        # Validaciones
        if len(X) == 0:
            self.logger.error("No hay características para entrenar")
            return None, None
        
        self.logger.info(f"\n✓ Características cargadas: {X.shape}")
        self.logger.info(f"  - Muestras: {X.shape[0]}")
        self.logger.info(f"  - Dimensiones: {X.shape[1]}")
        if self.feature_fusion:
            info = self.feature_fusion.get_feature_info()
            self.logger.info(f"  - HOG: {info['hog_dim']}")
            self.logger.info(f"  - MFCC: {info['mfcc_dim']}")
        
        # Crear modelo
        model = SVMModel(kernel='rbf', C=1.0, gamma='scale')
        
        if validate:
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=(1 - self.train_test_split_ratio),
                stratify=y,
                random_state=42
            )
            
            self.logger.info(f"\n📊 División Train/Test:")
            self.logger.info(f"  - Entrenamiento: {len(X_train)} muestras ({self.train_test_split_ratio:.0%})")
            self.logger.info(f"  - Prueba: {len(X_test)} muestras ({1-self.train_test_split_ratio:.0%})")
            self.logger.info(f"  - Clases: {len(set(y))}")
            
            # Entrenar
            self.logger.info(f"\n🔄 Entrenando SVM...")
            train_info = model.train(X_train, y_train)
            
            self.logger.info(f"\n✓ Entrenamiento completado en {train_info['training_time']:.2f}s")
            self.logger.info(f"  - Kernel: rbf")
            self.logger.info(f"  - C: 1.0")
            self.logger.info(f"  - Clases: {train_info['num_classes']}")
            
            # Evaluar
            self.logger.info(f"\n📈 Evaluando en datos de prueba...")
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            self.logger.info(f"\n" + "─" * 70)
            self.logger.info(f"MÉTRICAS DE EVALUACIÓN")
            self.logger.info(f"─" * 70)
            self.logger.info(f"  • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            self.logger.info(f"  • Precision: {metrics['precision_avg']:.4f}")
            self.logger.info(f"  • Recall:    {metrics['recall_avg']:.4f}")
            self.logger.info(f"  • F1-Score:  {metrics['f1_avg']:.4f}")
            self.logger.info(f"─" * 70)
            
            # Por clase
            self.logger.info(f"\nMÉTRICAS POR CLASE:")
            for cls in metrics['classes']:
                self.logger.info(f"\n  {cls}:")
                self.logger.info(f"    Precision: {metrics['precision_per_class'][cls]:.4f}")
                self.logger.info(f"    Recall:    {metrics['recall_per_class'][cls]:.4f}")
                self.logger.info(f"    F1-Score:  {metrics['f1_per_class'][cls]:.4f}")
                self.logger.info(f"    Muestras:  {metrics['support_per_class'][cls]}")
            
            self.logger.info("\n" + "=" * 70)
            
            return model, metrics
        
        else:
            # Entrenar con todos los datos
            self.logger.info(f"\n🔄 Entrenando SVM (sin validación)...")
            train_info = model.train(X, y)
            
            self.logger.info(f"\n✓ Entrenamiento completado en {train_info['training_time']:.2f}s")
            self.logger.info(f"  - Muestras totales: {train_info['num_samples']}")
            self.logger.info(f"  - Clases: {train_info['num_classes']}")
            
            return model, {'training_info': train_info}
