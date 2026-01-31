from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .svm_model import SVMModel
from .model_evaluator import ModelEvaluator

import numpy as np
from typing import Optional, Dict, Any

# Extractors y detectores
from feature_extraction.hog import HOGExtractor
from feature_extraction.mfcc import MFCCExtractor
from detection.audio_detection import AudioDetection


class ModelTrainer:
    """
    Entrenador para el modelo SVM con soporte para características HOG (imagen)
    y MFCC (audio). También puede validar audio usando AudioDetection antes
    de extraer MFCC.

    Parámetros opcionales:
      - use_hog (bool): extraer HOG si se proporcionan imágenes.
      - use_mfcc (bool): extraer MFCC si se proporcionan audios.
      - use_audio_detection (bool): validar audio con AudioDetection antes de MFCC.
    """
    
    def __init__(
        self,
        train_test_split_ratio: float = 0.8,
        use_hog: bool = False,
        use_mfcc: bool = False,
        use_audio_detection: bool = False,
        hog_params: Optional[Dict[str, Any]] = None,
        mfcc_params: Optional[Dict[str, Any]] = None,
        audio_params: Optional[Dict[str, Any]] = None
    ):
        self.train_test_split_ratio = train_test_split_ratio
        self.use_hog = use_hog
        self.use_mfcc = use_mfcc
        self.use_audio_detection = use_audio_detection

        self.hog_params = hog_params or {}
        self.mfcc_params = mfcc_params or {}
        self.audio_params = audio_params or {}

        # Instanciar extractores según sea necesario
        self.hog_extractor = HOGExtractor(**self.hog_params) if self.use_hog else None
        self.mfcc_extractor = MFCCExtractor(**self.mfcc_params) if self.use_mfcc else None
        self.audio_detector = AudioDetection(**self.audio_params) if self.use_audio_detection else None
    
    def _prepare_features(self, features):
        """
        Acepta:
          - np.ndarray ya calculado (se usa tal cual)
          - dict con claves 'images' (lista/ndarray de imágenes) y/o 'audios' (lista de arrays o rutas)
          - dict con claves 'hog' y/o 'mfcc' que ya contienen matrices de características

        Retorna una matriz numpy (N, D) lista para entrenar/evaluar.
        """
        # Si ya es numpy array, asumimos que es la matriz de características
        if isinstance(features, np.ndarray):
            return features

        if not isinstance(features, dict):
            raise TypeError('features debe ser np.ndarray o dict con imágenes/audios/características')

        hog_feats = None
        mfcc_feats = None

        # Si ya vienen características precalculadas
        if 'hog' in features:
            hog_feats = np.array(features['hog'], dtype=np.float32)
        if 'mfcc' in features:
            mfcc_feats = np.array(features['mfcc'], dtype=np.float32)

        # Extraer a partir de imágenes
        if hog_feats is None and self.use_hog and 'images' in features:
            images = features['images']
            if not isinstance(images, (list, np.ndarray)) or len(images) == 0:
                raise ValueError('images debe ser una lista/ndarray no vacía')
            hog_feats = self.hog_extractor.extract_batch(list(images))

        # Extraer a partir de audios
        if mfcc_feats is None and self.use_mfcc and 'audios' in features:
            audios = features['audios']
            if not isinstance(audios, (list, np.ndarray)) or len(audios) == 0:
                raise ValueError('audios debe ser una lista/ndarray no vacía')

            mfcc_list = []
            for idx, audio in enumerate(audios):
                # Si usamos detector, validar
                if self.audio_detector is not None:
                    # El detector espera ruta o array; se usa validate_audio_for_mfcc
                    try:
                        validation = self.audio_detector.validate_audio_for_mfcc(audio)
                        if not validation['is_valid']:
                            print(f"[ModelTrainer] ⚠️ Audio index {idx} no válido para MFCC: {validation['reason']}. Se usará padding de ceros.")
                            # Usar vector de ceros en caso de invalidación
                            mfcc_list.append(np.zeros(self.mfcc_extractor.n_mfcc * 2 if hasattr(self.mfcc_extractor, 'n_mfcc') else 13*2, dtype=np.float32))
                            continue
                    except Exception as e:
                        print(f"[ModelTrainer] ⚠️ Error validando audio index {idx}: {e}. Omitiendo...")
                        mfcc_list.append(np.zeros(self.mfcc_extractor.n_mfcc * 2 if hasattr(self.mfcc_extractor, 'n_mfcc') else 13*2, dtype=np.float32))
                        continue

                # Extraer estadísticas MFCC (vector tamaño fijo)
                try:
                    stats = self.mfcc_extractor.extract_statistics(audio)
                    mfcc_list.append(stats)
                except Exception as e:
                    print(f"[ModelTrainer] ⚠️ Error extrayendo MFCC index {idx}: {e}. Usando ceros como fallback.")
                    mfcc_list.append(np.zeros(self.mfcc_extractor.n_mfcc * 2 if hasattr(self.mfcc_extractor, 'n_mfcc') else 13*2, dtype=np.float32))

            mfcc_feats = np.vstack(mfcc_list).astype(np.float32)

        # Comprobar compatibilidad de tamaños
        N = None
        if hog_feats is not None:
            N = hog_feats.shape[0]
        if mfcc_feats is not None:
            if N is None:
                N = mfcc_feats.shape[0]
            elif mfcc_feats.shape[0] != N:
                raise ValueError('El número de muestras en HOG y MFCC no coincide')

        if hog_feats is not None and mfcc_feats is not None:
            # Concatenar características
            return np.concatenate([hog_feats, mfcc_feats], axis=1)
        elif hog_feats is not None:
            return hog_feats
        elif mfcc_feats is not None:
            return mfcc_feats

        raise ValueError('No se encontraron características para preparar. Verifique los parámetros y entradas.')
    
    def train(self, features, labels, validate=True):
        print(f"\n{'='*60}")
        print("ENTRENAMIENTO DEL MODELO SVM")
        print(f"{'='*60}")

        # Preparar características si es necesario
        X = self._prepare_features(features)

        model = SVMModel(kernel='rbf', C=1.0, gamma='scale')
        
        if validate:
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels,
                test_size=(1 - self.train_test_split_ratio),
                stratify=labels,
                random_state=42
            )
            
            print(f"\nDatos de entrenamiento: {len(X_train)}")
            print(f"Datos de prueba: {len(X_test)}")
            modalities = []
            if self.use_hog:
                modalities.append('HOG')
            if self.use_mfcc:
                modalities.append('MFCC')
            print(f"Usando modalidades: {', '.join(modalities) if modalities else 'features numéricas'}")

            train_info = model.train(X_train, y_train)
            
            print(f"\n✓ Entrenamiento completado en {train_info['training_time']:.2f}s")
            print(f"  - Muestras: {train_info['num_samples']}")
            print(f"  - Clases: {train_info['num_classes']}")
            
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            print(f"\nMétricas de evaluación:")
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - Precision (promedio): {metrics['precision_avg']:.4f}")
            print(f"  - Recall (promedio): {metrics['recall_avg']:.4f}")
            print(f"  - F1-Score (promedio): {metrics['f1_avg']:.4f}")
            
            return model, metrics
        else:
            train_info = model.train(X, labels)
            
            print(f"\n✓ Entrenamiento completado en {train_info['training_time']:.2f}s")
            print(f"  - Muestras totales: {train_info['num_samples']}")
            print(f"  - Clases: {train_info['num_classes']}")
            
            return model, {'training_info': train_info}
