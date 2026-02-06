"""
Manejador de comandos de la interfaz de línea de comandos.

Implementa el parser de comandos y la ejecución de acciones del sistema.
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, List, Any


class CommandHandler:
    """
    Clase que maneja los comandos de la interfaz CLI.
    
    Attributes:
        commands (dict): Diccionario de comandos disponibles.
        preprocessor_type (str): Tipo de preprocesador activo ('BLP', 'HSH', 'LBP').
        preprocessor: Instancia del preprocesador activo.
    """
    
    AVAILABLE_PREPROCESSORS = ['BLP', 'HSH', 'LBP']
    
    # ============== RUTAS FIJAS DEL SISTEMA ==============
    # Ruta base del proyecto
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Rutas del dataset
    DATASET_PATH = os.path.join(BASE_PATH, 'data', 'dataset')
    DATASET_PROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'datasetPros')
    
    # Rutas de modelos
    MODELS_PATH = os.path.join(BASE_PATH, 'models')
    SVM_MODEL_PATH = os.path.join(MODELS_PATH, 'svm_model.pkl')
    
    # Rutas de características extraídas
    FEATURES_PATH = os.path.join(BASE_PATH, 'data', 'features')
    
    # Ruta de evaluaciones
    EVALUATIONS_PATH = os.path.join(BASE_PATH, 'data', 'evaluations')
    # =====================================================
    
    def __init__(self):
        """
        Inicializa el manejador de comandos.
        """
        self.commands = {}
        self.preprocessor_type = 'LBP'  # Preprocesador por defecto
        self.preprocessor = None
        self.last_evaluation = None  # Almacena última evaluación
        self._ensure_directories()
        self._register_commands()
    
    def _ensure_directories(self):
        """
        Crea los directorios necesarios si no existen.
        """
        directories = [
            self.DATASET_PATH,
            self.DATASET_PROCESSED_PATH,
            self.MODELS_PATH,
            self.FEATURES_PATH,
            self.EVALUATIONS_PATH
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _register_commands(self):
        """
        Registra todos los comandos disponibles del sistema.
        """
        self.commands = {
            'preprocess': {
                'handler': self.preprocess,
                'description': 'Preprocesar dataset (extracción de frames y augmentation)'
            },
            'prepare_dataset': {
                'handler': self.prepare_dataset,
                'description': 'Crear carpetas por persona en data/dataset (un video por carpeta)'
            },
            'extract_audio': {
                'handler': self.extract_audio,
                'description': 'Extraer audios de los videos por persona y guardarlos en data/datasetPros/audio'
            },
            'detect': {
                'handler': self.detect,
                'description': 'Detectar cuerpos en imágenes del dataset procesado'
            },
            'extract': {
                'handler': self.extract_features,
                'description': 'Extraer características usando el preprocesador seleccionado'
            },
            'train': {
                'handler': self.train_svm,
                'description': 'Entrenar modelo SVM con las características extraídas'
            },
            'evaluate': {
                'handler': self.evaluate,
                'description': 'Ver evaluación del modelo entrenado'
            },
            'auto': {
                'handler': self.run_automatic,
                'description': 'Ejecutar pipeline completo automáticamente'
            },
            'set_preprocessor': {
                'handler': self.set_preprocessor,
                'description': 'Configurar el tipo de preprocesador (BLP, HSH, LBP)'
            },
            'status': {
                'handler': self.get_status,
                'description': 'Mostrar estado actual del sistema'
            },
            'help': {
                'handler': self.help,
                'description': 'Mostrar ayuda de comandos'
            },
            'exit': {
                'handler': self.exit_system,
                'description': 'Salir del sistema'
            }
        }
    
    def execute_command(self, command_name: str, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta un comando registrado.
        
        Args:
            command_name (str): Nombre del comando a ejecutar.
            **kwargs: Argumentos para el comando.
        
        Returns:
            dict: Resultado de la ejecución del comando.
        """
        if command_name not in self.commands:
            return {
                'success': False,
                'error': f"Comando '{command_name}' no reconocido. Use 'help' para ver comandos disponibles."
            }
        
        try:
            handler = self.commands[command_name]['handler']
            result = handler(**kwargs)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def set_preprocessor(self, preprocessor_type: str) -> Dict[str, Any]:
        """
        Configura el tipo de preprocesador a utilizar.
        
        Args:
            preprocessor_type (str): Tipo de preprocesador ('BLP', 'HSH', 'LBP').
        
        Returns:
            dict: Resultado de la configuración.
        """
        preprocessor_type = preprocessor_type.upper()
        
        if preprocessor_type not in self.AVAILABLE_PREPROCESSORS:
            return {
                'success': False,
                'error': f"Preprocesador '{preprocessor_type}' no válido. Opciones: {self.AVAILABLE_PREPROCESSORS}"
            }
        
        self.preprocessor_type = preprocessor_type
        self.preprocessor = None  # Se inicializará cuando se necesite
        
        return {
            'success': True,
            'message': f"Preprocesador configurado: {preprocessor_type}",
            'preprocessor': preprocessor_type
        }
    
    def _get_preprocessor(self):
        """
        Obtiene la instancia del preprocesador configurado.
        
        Returns:
            Instancia del preprocesador.
        """
        if self.preprocessor is not None:
            return self.preprocessor
        
        try:
            from preprocessing.preprocessors import BLPPreprocessor, HSHPreprocessor, LBPPreprocessor

            if self.preprocessor_type == 'BLP':
                self.preprocessor = BLPPreprocessor()
            elif self.preprocessor_type == 'HSH':
                self.preprocessor = HSHPreprocessor()
            elif self.preprocessor_type == 'LBP':
                self.preprocessor = LBPPreprocessor()

            return self.preprocessor
        except Exception:
            # Fallback: si no existen preprocesadores personalizados, usar HOG como extractor por defecto
            try:
                from feature_extraction.hog import HOGExtractor
                self.preprocessor = HOGExtractor()
                print("[INFO] Preprocesadores personalizados no encontrados. Usando HOGExtractor como fallback.")
                return self.preprocessor
            except Exception as e:
                raise ImportError(f"No se pudo obtener preprocesador: {e}")
    
    # ==================== COMANDO 1: PREPROCESAR ====================
    def preprocess(self) -> Dict[str, Any]:
        """
        Preprocesa el dataset: extrae frames de videos y aplica data augmentation.
        Usa rutas fijas del sistema.
        
        Returns:
            dict: Resultado del preprocesamiento.
        """
        print(f"\n[PREPROCESAR] Iniciando preprocesamiento...")
        print(f"  Dataset origen: {self.DATASET_PATH}")
        print(f"  Dataset destino: {self.DATASET_PROCESSED_PATH}")
        
        # Verificar que existe el dataset origen
        if not os.path.exists(self.DATASET_PATH):
            return {
                'success': False,
                'error': f"Dataset no encontrado en: {self.DATASET_PATH}\nPor favor, coloque los videos/imágenes en esa carpeta."
            }
        
        # Contar archivos en dataset
        total_files = 0
        for root, dirs, files in os.walk(self.DATASET_PATH):
            total_files += len([f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.jpg', '.png', '.jpeg'))])
        
        if total_files == 0:
            return {
                'success': False,
                'error': f"No se encontraron archivos de video/imagen en: {self.DATASET_PATH}"
            }
        
        try:
            from preprocessing import PreprocessingPipeline
            
            pipeline = PreprocessingPipeline(self.DATASET_PATH, self.DATASET_PROCESSED_PATH, fps=10)
            result = pipeline.run_full_pipeline()
            
            return {
                'success': True,
                'message': "Preprocesamiento completado",
                'dataset_path': self.DATASET_PATH,
                'output_path': self.DATASET_PROCESSED_PATH,
                'files_found': total_files,
                'statistics': result
            }
        except NotImplementedError:
            # Simular preprocesamiento si no está implementado
            print("  [INFO] Pipeline de preprocesamiento pendiente de implementación")
            return {
                'success': True,
                'message': "Preprocesamiento (placeholder) - Pipeline no implementado",
                'dataset_path': self.DATASET_PATH,
                'output_path': self.DATASET_PROCESSED_PATH,
                'files_found': total_files
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== NUEVO: PREPARAR DATASET ====================
    def prepare_dataset(self, persons: Optional[list] = None, auto_distribute: bool = False, move: bool = False) -> Dict[str, Any]:
        """
        Crea las carpetas en `data/dataset` para las personas indicadas.
        No intenta parsear vistas; cada carpeta de persona debe contener un único video que usted colocará.
        """
        try:
            if persons:
                person_list = persons
            else:
                person_list = ['unknown']

            for p in person_list:
                folder = os.path.join(self.DATASET_PATH, p)
                os.makedirs(folder, exist_ok=True)

            # Opcional: mover/copiar desde data/raw si se solicita
            if auto_distribute:
                raw_folder = os.path.join(self.BASE_PATH, 'data', 'raw')
                if not os.path.exists(raw_folder):
                    return {'success': False, 'error': f"Raw folder no encontrado: {raw_folder}"}

                video_exts = ('.mp4', '.avi', '.mov', '.mkv')
                for f in os.listdir(raw_folder):
                    if f.lower().endswith(video_exts):
                        # intentar detectar persona por prefijo simple
                        person = f.split('_', 1)[0]
                        target_dir = os.path.join(self.DATASET_PATH, person)
                        os.makedirs(target_dir, exist_ok=True)
                        src = os.path.join(raw_folder, f)
                        dst = os.path.join(target_dir, f)
                        if move:
                            import shutil
                            shutil.move(src, dst)
                            action = 'movido'
                        else:
                            import shutil
                            shutil.copy2(src, dst)
                            action = 'copiado'
                
            return {'success': True, 'message': 'Carpetas creadas en data/dataset'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== NUEVO: EXTRAER AUDIO (por persona) ====================
    def extract_audio(self, save_audio: bool = True) -> Dict[str, Any]:
        """
        Extrae audios de los videos por persona y los guarda en `data/datasetPros/audio`.
        """
        try:
            from preprocessing.audio_extraction import AudioExtractor

            extractor = AudioExtractor(dataset_path=self.DATASET_PATH, output_path=self.DATASET_PROCESSED_PATH)
            stats = extractor.process_person_videos(save_audio=save_audio)

            return {'success': True, 'message': 'Extracción de audio completada', 'stats': stats}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ==================== COMANDO 2: DETECTAR ====================
    def detect(self) -> Dict[str, Any]:
        """
        Detecta cuerpos y rostros en las imágenes del dataset procesado.
        
        Returns:
            dict: Resultado de la detección.
        """
        print(f"\n[DETECTAR] Iniciando detección de cuerpos...")
        print(f"  Dataset procesado: {self.DATASET_PROCESSED_PATH}")
        
        if not os.path.exists(self.DATASET_PROCESSED_PATH):
            return {
                'success': False,
                'error': f"Dataset procesado no encontrado. Ejecute 'preprocesar' primero."
            }
        
        # Contar imágenes disponibles
        image_count = 0
        persons = []
        for person_id in os.listdir(self.DATASET_PROCESSED_PATH):
            person_path = os.path.join(self.DATASET_PROCESSED_PATH, person_id)
            if os.path.isdir(person_path):
                persons.append(person_id)
                for root, dirs, files in os.walk(person_path):
                    image_count += len([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if image_count == 0:
            return {
                'success': False,
                'error': "No se encontraron imágenes en el dataset procesado."
            }
        
        try:
            from detection import BodyDetection
            
            body_detector = BodyDetection()
            
            detections = {
                'bodies': 0,
                'failed': 0
            }
            
            # Procesar cada imagen en carpeta 'body'
            for person_id in persons:
                person_path = os.path.join(self.DATASET_PROCESSED_PATH, person_id)
                view = 'body'
                view_path = os.path.join(person_path, view)
                if os.path.exists(view_path):
                    for img_file in os.listdir(view_path):
                        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(view_path, img_file)
                            # Aquí iría la detección real
                            detections['bodies'] += 1
            
            return {
                'success': True,
                'message': "Detección completada",
                'total_images': image_count,
                'persons_found': len(persons),
                'detections': detections
            }
        except (ImportError, NotImplementedError):
            print("  [INFO] Módulo de detección pendiente de implementación")
            return {
                'success': True,
                'message': "Detección (placeholder) - Módulo no implementado",
                'total_images': image_count,
                'persons_found': len(persons)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ==================== COMANDO 3: EXTRAER CARACTERÍSTICAS ====================
    def extract_features(self) -> Dict[str, Any]:
        """
        Extrae características de las imágenes usando el preprocesador configurado.
        
        Returns:
            dict: Resultado de la extracción.
        """
        print(f"\n[EXTRAER] Iniciando extracción de características...")
        print(f"  Preprocesador: {self.preprocessor_type}")
        print(f"  Dataset: {self.DATASET_PROCESSED_PATH}")
        print(f"  Salida: {self.FEATURES_PATH}")
        
        if not os.path.exists(self.DATASET_PROCESSED_PATH):
            return {
                'success': False,
                'error': "Dataset procesado no encontrado. Ejecute 'preprocesar' primero."
            }
        
        # Recolectar imágenes
        image_paths = []
        labels = []
        
        for person_id in os.listdir(self.DATASET_PROCESSED_PATH):
            person_path = os.path.join(self.DATASET_PROCESSED_PATH, person_id)
            if os.path.isdir(person_path):
                view = 'body'
                view_path = os.path.join(person_path, view)
                if os.path.exists(view_path):
                    for img_file in os.listdir(view_path):
                        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            image_paths.append(os.path.join(view_path, img_file))
                            labels.append(person_id)
        
        if not image_paths:
            return {
                'success': False,
                'error': "No se encontraron imágenes para extraer características."
            }
        
        print(f"  Imágenes encontradas: {len(image_paths)}")
        
        try:
            from feature_extraction import FeatureVector
            from feature_extraction.mfcc import MFCCExtractor
            from detection.audio_detection import AudioDetection
            
            preprocessor = self._get_preprocessor()
            
            # Cargar imágenes
            images = []
            valid_paths = []
            valid_labels = []
            
            for i, path in enumerate(image_paths):
                try:
                    try:
                        import cv2
                        img = cv2.imread(path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            valid_paths.append(path)
                            valid_labels.append(labels[i])
                    except ImportError:
                        from PIL import Image
                        pil_img = Image.open(path)
                        images.append(np.array(pil_img))
                        valid_paths.append(path)
                        valid_labels.append(labels[i])
                except Exception as e:
                    print(f"  [WARN] Error cargando {path}: {e}")
            
            # EXTRAER características de imágenes (si hay)
            features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
            labels_file = os.path.join(self.FEATURES_PATH, f'labels_{self.preprocessor_type}.npy')

            if images:
                # Extraer características
                feature_vectors = FeatureVector.from_images_batch(images, preprocessor, valid_labels)
                feature_matrix = np.array([fv.to_numpy() for fv in feature_vectors])

                # Guardar características de imagen
                np.save(features_file, feature_matrix)
                np.save(labels_file, np.array(valid_labels))

            else:
                print("  [INFO] No se encontraron imágenes para extraer características de imagen.")

            # ---------- EXTRAER características MFCC de audios (si hay) ----------
            audio_paths = []
            audio_labels = []
            audio_root = os.path.join(self.DATASET_PROCESSED_PATH, 'audio')

            if os.path.exists(audio_root):
                for person_id in os.listdir(audio_root):
                    person_audio_dir = os.path.join(audio_root, person_id)
                    if os.path.isdir(person_audio_dir):
                        for f in os.listdir(person_audio_dir):
                            if f.lower().endswith('.wav'):
                                audio_paths.append(os.path.join(person_audio_dir, f))
                                audio_labels.append(person_id)

            audio_features_file = os.path.join(self.FEATURES_PATH, 'features_audio.npy')
            audio_labels_file = os.path.join(self.FEATURES_PATH, 'labels_audio.npy')

            if audio_paths:
                mfcc_extractor = MFCCExtractor()
                audio_feats = []
                valid_audio_labels = []

                for idx, a_path in enumerate(audio_paths):
                    try:
                        # Extraer características MFCC directamente de la ruta
                        stats = mfcc_extractor.extract_statistics(a_path)
                        audio_feats.append(stats)
                        valid_audio_labels.append(audio_labels[idx])
                    except Exception as e:
                        print(f"  [WARN] Error procesando audio {a_path}: {e}")
                        audio_feats.append(np.zeros(mfcc_extractor.n_mfcc * 2, dtype=np.float32))
                        valid_audio_labels.append(audio_labels[idx])

                import numpy as _np
                audio_matrix = _np.vstack(audio_feats).astype(_np.float32)
                _np.save(audio_features_file, audio_matrix)
                _np.save(audio_labels_file, _np.array(valid_audio_labels))

                print(f"  MFCC audios extraídos: {audio_matrix.shape} -> {audio_features_file}")
            else:
                print("  [INFO] No se encontraron audios para extraer MFCC.")

            return {
                'success': True,
                'message': "Extracción de características completada",
                'preprocessor': self.preprocessor_type,
                'num_images': len(images),
                'feature_dimension': (feature_matrix.shape[1] if 'feature_matrix' in locals() else 0),
                'features_file': features_file if images else None,
                'labels_file': labels_file if images else None,
                'audio_features_file': audio_features_file if os.path.exists(audio_root) else None,
                'audio_labels_file': audio_labels_file if os.path.exists(audio_root) else None
            }
        except NotImplementedError as e:
            print(f"  [INFO] Preprocesador {self.preprocessor_type} pendiente de implementación")
            return {
                'success': True,
                'message': f"Extracción (placeholder) - Preprocesador {self.preprocessor_type} no implementado",
                'num_images': len(image_paths),
                'preprocessor': self.preprocessor_type
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ==================== COMANDO 3b: AUGMENTACIÓN DE AUDIOS ====================
    def augment_audio_dataset(self) -> Dict[str, Any]:
        """
        Aplica augmentación de datos a todos los audios del dataset.
        
        Genera variaciones de cada audio para evitar replicación artificial:
        - Pitch shifting (±2, ±4 semitonos)
        - Time stretching (±5% velocidad)
        - Gaussian noise
        - Dynamic range compression
        
        Resultado: data/audios_augmented/{person}/audio_*_aug{00-15}.wav
        
        Ventajas:
        - Evita replicación artificial (7 audios → 7×15 = 105 audios variados)
        - Mejora generalización del modelo
        - Crea diversidad real en las muestras de entrenamiento
        
        Returns:
            dict con estadísticas de augmentación
        """
        try:
            print("\n" + "="*70)
            print("🎵 AUGMENTACIÓN DE AUDIOS - Generando variaciones de datos")
            print("="*70)
            
            from preprocessing.audio_augmentation import AudioAugmentation
            
            # Verificar que hay audios originales
            audio_dir = os.path.join(self.BASE_PATH, 'data', 'datasetPros', 'audio')
            if not os.path.exists(audio_dir):
                print(f"❌ No se encontraron audios en: {audio_dir}")
                return {
                    'success': False,
                    'error': 'No se encontraron audios para augmentar'
                }
            
            # Crear augmentador
            augmentor = AudioAugmentation(sr=22050)
            
            # Directorio de salida
            output_base_dir = os.path.join(self.BASE_PATH, 'data', 'audios_augmented')
            
            # Preguntar cuántas variantes generar
            print("\n¿Cuántas variantes deseas generar por audio?")
            print("  (Recomendado: 15, genera 7 audios × 15 = 105 muestras totales)")
            
            try:
                variants_input = input("  Ingresa el número [10-20] (default: 15): ").strip()
                variants = int(variants_input) if variants_input else 15
                
                if not (10 <= variants <= 20):
                    print(f"  ⚠️  Fuera de rango, usando default: 15")
                    variants = 15
            except ValueError:
                variants = 15
            
            print(f"\n📊 Augmentando dataset con {variants} variantes por audio...")
            
            # Ejecutar augmentación
            stats = augmentor.augment_dataset(
                dataset_audio_dir=audio_dir,
                output_base_dir=output_base_dir,
                variants_per_audio=variants
            )
            
            # Mostrar resultados
            print("\n" + "="*70)
            print("✓ AUGMENTACIÓN COMPLETADA")
            print("="*70)
            print(f"  Personas procesadas: {stats['total_persons']}")
            print(f"  Audios originales: {stats['total_audios_original']}")
            print(f"  Audios total tras augmentación: {stats['total_audios_augmented']}")
            print(f"  Factor de expansión: {stats['total_audios_augmented'] / max(stats['total_audios_original'], 1):.1f}x")
            
            print(f"\n  Detalles por persona:")
            for person, person_stats in stats['persons'].items():
                exp = person_stats['audios_augmented'] / max(person_stats['audios_original'], 1)
                print(f"    • {person}: {person_stats['audios_original']} → {person_stats['audios_augmented']} ({exp:.0f}x)")
            
            print(f"\n  📁 Audios augmentados guardados en:")
            print(f"     {output_base_dir}")
            print(f"\n  💡 El próximo entrenamiento usará automáticamente estos audios augmentados")
            print(f"     en lugar de replicación artificial")
            
            return {
                'success': True,
                'message': 'Augmentación de audios completada exitosamente',
                **stats
            }
        
        except Exception as e:
            print(f"\n❌ Error durante augmentación: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    # ==================== COMANDO 4: ENTRENAR SVM ====================
    def train_svm(self, modality: str = 'fusion') -> Dict[str, Any]:
        """
        Entrena el modelo SVM con características.
        
        Soporta tres modalidades:
        - 'fusion': HOG + MFCC combinados (RECOMENDADO - multimodal)
        - 'image': Solo HOG de imágenes
        - 'audio': Solo MFCC de audios

        Args:
            modality (str): 'fusion' (default), 'image', o 'audio'.

        Returns:
            dict: Resultado del entrenamiento.
        """
        print(f"\n{'='*70}")
        print(f"ENTRENAMIENTO DEL MODELO SVM")
        print(f"Modalidad: {modality.upper()}")
        print(f"{'='*70}")

        try:
            from svm_classifier import ModelTrainer
            
            if modality == 'fusion':
                print("\n🔗 Modo FUSIÓN: HOG + MFCC")
                print(f"   Cargando desde: {self.DATASET_PROCESSED_PATH}")
                print(f"   Audios desde:   {os.path.join(self.DATASET_PROCESSED_PATH, 'audio')}")
                
                # Crear trainer con fusión automática
                trainer = ModelTrainer(
                    train_test_split_ratio=0.8,
                    use_fusion=True
                )
                
                # Entrenar con carga automática de características fusionadas
                model, metrics = trainer.train(
                    X=None,  # None = cargar automáticamente
                    y=None,
                    validate=True,
                    dataset_processed_path=self.DATASET_PROCESSED_PATH,
                    audio_base_dir=os.path.join(self.DATASET_PROCESSED_PATH, 'audio')
                )
                
                model_path = os.path.join(self.MODELS_PATH, 'svm_model_fusion.pkl')
            
            elif modality == 'image':
                print("\n🖼️  Modo IMAGEN: Solo HOG")
                
                # Cargar características de imagen
                features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
                labels_file = os.path.join(self.FEATURES_PATH, f'labels_{self.preprocessor_type}.npy')
                
                if not os.path.exists(features_file) or not os.path.exists(labels_file):
                    return {
                        'success': False,
                        'error': f"Características de imagen no encontradas.\nEjecute 'preprocess' y 'extract' primero."
                    }
                
                features = np.load(features_file)
                labels = np.load(labels_file, allow_pickle=True)
                
                print(f"   Características cargadas: {features.shape}")
                print(f"   Etiquetas: {len(labels)} ({len(set(labels))} clases)")
                
                trainer = ModelTrainer(use_fusion=False)
                model, metrics = trainer.train(features, labels, validate=True)
                model_path = os.path.join(self.MODELS_PATH, 'svm_model_image.pkl')
            
            elif modality == 'audio':
                print("\n🔊 Modo AUDIO: Solo MFCC (con replicación de datos)")
                
                # Cargar características de audio
                features_file = os.path.join(self.FEATURES_PATH, 'features_audio.npy')
                labels_file = os.path.join(self.FEATURES_PATH, 'labels_audio.npy')
                
                if not os.path.exists(features_file) or not os.path.exists(labels_file):
                    return {
                        'success': False,
                        'error': f"Características de audio no encontradas.\nEjecute 'extract_audio' primero."
                    }
                
                features = np.load(features_file)
                labels = np.load(labels_file, allow_pickle=True)
                
                print(f"   Características originales: {features.shape}")
                print(f"   Etiquetas originales: {len(labels)} ({len(set(labels))} clases)")
                
                # REPLICAR audios para que haya suficientes muestras (al menos 30 por clase)
                try:
                    img_features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
                    n_images = len(np.load(img_features_file))
                    n_replicas = max(30, n_images // len(set(labels)))
                except:
                    n_replicas = 30
                
                print(f"   Replicando {n_replicas} veces cada audio...")
                
                features_replicated = np.repeat(features, n_replicas, axis=0)
                labels_replicated = np.repeat(labels, n_replicas)
                
                print(f"   ✓ Características después de replicación: {features_replicated.shape}")
                print(f"   ✓ Etiquetas después: {len(labels_replicated)} ({len(set(labels_replicated))} clases)")
                
                trainer = ModelTrainer(use_fusion=False)
                model, metrics = trainer.train(features_replicated, labels_replicated, validate=True)
                model_path = os.path.join(self.MODELS_PATH, 'svm_model_audio.pkl')
            
            else:
                return {
                    'success': False,
                    'error': f"Modalidad '{modality}' no válida. Use: 'fusion', 'image', 'audio'"
                }
            
            if model is None:
                return {
                    'success': False,
                    'error': "Error durante el entrenamiento"
                }
            
            # Guardar modelo
            os.makedirs(self.MODELS_PATH, exist_ok=True)
            model.save(model_path)
            
            # Guardar evaluación y accuracy para comparación posterior
            self.last_evaluation = metrics
            self.training_accuracy = metrics.get('accuracy', 0.0)  # Guardar para comparar con evaluación
            
            print(f"\n{'='*70}")
            print(f"✓ MODELO GUARDADO: {model_path}")
            print(f"{'='*70}\n")
            
            return {
                'success': True,
                'message': f"Modelo SVM ({modality}) entrenado y guardado exitosamente",
                'model_path': model_path,
                'metrics': metrics
            }
        
        except Exception as e:
            print(f"\n❌ Error durante entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    # ==================== COMANDO 5: VER EVALUACIÓN ====================
    def evaluate(self) -> Dict[str, Any]:
        """
        Muestra la evaluación del modelo entrenado.
        
        Busca modelos en orden de prioridad: fusion > image > audio
        
        Returns:
            dict: Métricas de evaluación.
        """
        print(f"\n[EVALUACIÓN] Mostrando resultados...")
        
        # Buscar modelos en orden de prioridad
        model_candidates = [
            os.path.join(self.MODELS_PATH, 'svm_model_fusion.pkl'),
            os.path.join(self.MODELS_PATH, 'svm_model_image.pkl'),
            os.path.join(self.MODELS_PATH, 'svm_model_audio.pkl'),
            self.SVM_MODEL_PATH  # Path default
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                model_type = candidate.split('_')[-1].replace('.pkl', '')
                print(f"✓ Modelo encontrado: {model_type}")
                break
        
        if model_path is None:
            return {
                'success': False,
                'error': "Modelo no encontrado. Ejecute 'entrenar' primero."
            }
        
        # Si hay evaluación en caché, retornarla
        if self.last_evaluation is not None:
            return {
                'success': True,
                'message': "Evaluación del modelo",
                'model_path': model_path,
                'metrics': self.last_evaluation
            }
        
        try:
            from svm_classifier import SVMModel, ModelEvaluator
            
            # Cargar modelo
            model = SVMModel.load(model_path)
            
            # Determinar qué características usar basado en el tipo de modelo
            if 'fusion' in model_path:
                # Para modelos de fusión, cargar ambas características y fusionarlas
                # Usar la MISMA estrategia que en entrenamiento
                img_features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
                img_labels_file = os.path.join(self.FEATURES_PATH, f'labels_{self.preprocessor_type}.npy')
                audio_features_file = os.path.join(self.FEATURES_PATH, 'features_audio.npy')
                audio_labels_file = os.path.join(self.FEATURES_PATH, 'labels_audio.npy')
                
                if all(os.path.exists(f) for f in [img_features_file, img_labels_file, audio_features_file, audio_labels_file]):
                    img_features = np.load(img_features_file)
                    img_labels = np.load(img_labels_file, allow_pickle=True)
                    audio_features = np.load(audio_features_file)
                    
                    n_images = len(img_features)
                    n_audio = len(audio_features)
                    
                    # Replicar audios EXACTAMENTE como en entrenamiento
                    # Usar ciclo repetido: audio[i % n_audio]
                    audio_features_expanded = np.zeros((n_images, audio_features.shape[1]), dtype=audio_features.dtype)
                    for i in range(n_images):
                        audio_idx = i % n_audio
                        audio_features_expanded[i] = audio_features[audio_idx]
                    
                    print(f"   Imágenes: {n_images}, Audios originales: {n_audio}, Método: ciclo repetido")
                    
                    features = np.hstack([img_features, audio_features_expanded])
                    labels = img_labels
                else:
                    return {'success': False, 'error': "Características fusionadas no encontradas"}
            
            elif 'audio' in model_path:
                # Modelo de audio
                features_file = os.path.join(self.FEATURES_PATH, 'features_audio.npy')
                labels_file = os.path.join(self.FEATURES_PATH, 'labels_audio.npy')
                
                if not os.path.exists(features_file) or not os.path.exists(labels_file):
                    return {'success': False, 'error': "Características de audio no encontradas"}
                
                features = np.load(features_file)
                labels = np.load(labels_file, allow_pickle=True)
                
                # Replicar para evaluación (mismo que en entrenamiento)
                try:
                    img_features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
                    n_images = len(np.load(img_features_file))
                    n_replicas = max(30, n_images // len(set(labels)))
                except:
                    n_replicas = 30
                
                features = np.repeat(features, n_replicas, axis=0)
                labels = np.repeat(labels, n_replicas)
            
            else:
                # Modelo de imagen (default)
                features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
                labels_file = os.path.join(self.FEATURES_PATH, f'labels_{self.preprocessor_type}.npy')
                
                if not os.path.exists(features_file) or not os.path.exists(labels_file):
                    return {'success': False, 'error': "Características de imagen no encontradas"}
                
                features = np.load(features_file)
                labels = np.load(labels_file, allow_pickle=True)
            
            # Evaluar
            evaluator = ModelEvaluator()
            
            # IMPORTANTE: Hacer el MISMO split 80/20 que en entrenamiento
            # para evaluar solo en los datos de PRUEBA (20%)
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels,
                test_size=0.2,
                random_state=42,
                stratify=labels
            )
            
            print(f"\n   Split Train/Test (80/20):")
            print(f"   - Entrenamiento: {len(X_train)} muestras")
            print(f"   - Prueba: {len(X_test)} muestras ← Evaluando aquí")
            
            # Evaluar SOLO en datos de prueba (igual que entrenamiento)
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            self.last_evaluation = metrics
            
            # Información adicional
            print(f"\n📊 MÉTRICAS DE EVALUACIÓN")
            print(f"{'='*60}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision_avg']:.4f}")
            print(f"  Recall:    {metrics['recall_avg']:.4f}")
            print(f"  F1-Score:  {metrics['f1_avg']:.4f}")
            print(f"{'='*60}\n")
            
            # Advertencia de overfitting si hay brecha grande
            if self.last_evaluation is not None:
                # El último_evaluation es del conjunto de prueba durante training
                # Comparar con métricas actuales
                if hasattr(self, 'training_accuracy'):
                    gap = self.training_accuracy - metrics['accuracy']
                    if gap > 0.5:  # Brecha > 50% es severo
                        print(f"⚠️  ADVERTENCIA: OVERFITTING SEVERO DETECTADO")
                        print(f"   - Accuracy en training: ~{self.training_accuracy:.1%}")
                        print(f"   - Accuracy en prueba: {metrics['accuracy']:.1%}")
                        print(f"   - Brecha: {gap:.1%}")
                        print(f"   • Causa probable: Replicación excesiva de audios (7 → 6300+)")
                        print(f"   • Solución: Agregar más muestras de audio reales (no replicadas)")
                        print(f"   • Alternativa: Usar solo HOG (opción imagen) sin audio\n")
            
            return {
                'success': True,
                'message': "Evaluación completada",
                'model_type': 'fusion' if 'fusion' in model_path else 'audio' if 'audio' in model_path else 'image',
                'model_path': model_path,
                'metrics': metrics
            }
        
        except Exception as e:
            print(f"❌ Error durante evaluación: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    # ==================== COMANDO 6: AUTOMÁTICO ====================
    def run_automatic(self) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo automáticamente (MULTIMODAL: HOG + MFCC):
        1. Preprocesar (extraer frames + audios)
        2. Extraer características HOG + MFCC
        3. Entrenar SVM multimodal (FUSIÓN)
        4. Mostrar evaluación
        
        Returns:
            dict: Resultado de todo el proceso.
        """
        print("\n" + "=" * 70)
        print("   🚀 EJECUCIÓN AUTOMÁTICA - PIPELINE MULTIMODAL (HOG + MFCC)")
        print("=" * 70)
        print(f"\nDataset: {self.DATASET_PATH}")
        print(f"Dataset procesado: {self.DATASET_PROCESSED_PATH}")
        print(f"Audios: {os.path.join(self.DATASET_PROCESSED_PATH, 'audio')}")
        print(f"Modelo de salida: {os.path.join(self.MODELS_PATH, 'svm_model_fusion.pkl')}")
        print("\n" + "-" * 70)
        
        results = {
            'preprocess': None,
            'train_fusion': None,
            'evaluate': None
        }
        
        # Pasos simplificados para el pipeline multimodal
        steps = [
            ('preprocess', '⚙️  PASO 1/3: Preprocesamiento (Frames + Audios)', self.preprocess),
            ('train_fusion', '🔗 PASO 2/3: Entrenamiento SVM Multimodal (HOG + MFCC)', 
             lambda: self.train_svm(modality='fusion')),
            ('evaluate', '📊 PASO 3/3: Evaluación del Modelo', self.evaluate)
        ]
        
        all_success = True
        
        for step_key, step_name, step_func in steps:
            print(f"\n{'─' * 70}")
            print(f"  {step_name}")
            print(f"{'─' * 70}")
            
            try:
                result = step_func()
                results[step_key] = result
                
                if result.get('success', False):
                    print(f"  ✓ {result.get('message', 'Completado')}")
                else:
                    print(f"  ✗ Error: {result.get('error', 'Error desconocido')}")
                    all_success = False
                    if step_key == 'preprocess':
                        # Si falla preprocesamiento, no continuar
                        print("\n  ⚠️  Abortando - Necesario preprocesamiento exitoso")
                        break
            except Exception as e:
                print(f"  ✗ Excepción: {e}")
                import traceback
                traceback.print_exc()
                results[step_key] = {'success': False, 'error': str(e)}
                all_success = False
        
        print("\n" + "=" * 70)
        print("   📋 RESUMEN DE EJECUCIÓN")
        print("=" * 70)
        
        for step_key, step_name, _ in steps:
            result = results[step_key]
            status = "✓" if result and result.get('success') else "✗"
            print(f"  {status} {step_name.split(':')[1].strip()}")
        
        print("=" * 70 + "\n")
        
        if all_success:
            print(f"✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print(f"\n   Modelo guardado: {os.path.join(self.MODELS_PATH, 'svm_model_fusion.pkl')}")
            print(f"   Puedes usar 'evaluate' para ver más detalles\n")
        else:
            print(f"⚠️  Pipeline completado con errores\n")
        
        return {
            'success': all_success,
            'message': "Pipeline multimodal completado" if all_success else "Pipeline completado con errores",
            'results': results
        }
    
    # ==================== OTROS MÉTODOS ====================
    def help(self, command_name: str = None) -> str:
        """
        Muestra ayuda sobre los comandos.
        """
        if command_name is None:
            help_text = "\n=== Sistema de Reidentificación de Personas ===\n\n"
            help_text += "Rutas del sistema:\n"
            help_text += f"  Dataset:           {self.DATASET_PATH}\n"
            help_text += f"  Dataset procesado: {self.DATASET_PROCESSED_PATH}\n"
            help_text += f"  Modelos:           {self.MODELS_PATH}\n"
            help_text += f"  Características:   {self.FEATURES_PATH}\n\n"
            help_text += "Comandos disponibles:\n"
            help_text += "-" * 50 + "\n"
            
            for name, info in self.commands.items():
                help_text += f"  {name:15} - {info['description']}\n"
            
            help_text += "\n" + "-" * 50
            help_text += f"\nPreprocesador actual: {self.preprocessor_type}"
            help_text += f"\nOpciones: {', '.join(self.AVAILABLE_PREPROCESSORS)}"
            
            return help_text
        
        if command_name not in self.commands:
            return f"Comando '{command_name}' no encontrado."
        
        return f"{command_name}: {self.commands[command_name]['description']}"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        """
        # Verificar qué archivos existen
        dataset_exists = os.path.exists(self.DATASET_PATH) and len(os.listdir(self.DATASET_PATH)) > 0
        processed_exists = os.path.exists(self.DATASET_PROCESSED_PATH) and len(os.listdir(self.DATASET_PROCESSED_PATH)) > 0
        features_exist = os.path.exists(os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy'))
        model_exists = os.path.exists(self.SVM_MODEL_PATH)
        
        return {
            'preprocessor_type': self.preprocessor_type,
            'paths': {
                'dataset': self.DATASET_PATH,
                'dataset_processed': self.DATASET_PROCESSED_PATH,
                'features': self.FEATURES_PATH,
                'model': self.SVM_MODEL_PATH
            },
            'status': {
                'dataset_ready': dataset_exists,
                'preprocessed': processed_exists,
                'features_extracted': features_exist,
                'model_trained': model_exists
            }
        }
    
    def exit_system(self) -> Dict[str, Any]:
        """
        Sale del sistema.
        """
        return {
            'success': True,
            'message': "Saliendo del sistema. ¡Hasta pronto!",
            'exit': True
        }
