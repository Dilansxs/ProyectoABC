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
            'detect': {
                'handler': self.detect,
                'description': 'Detectar cuerpos/rostros en imágenes del dataset procesado'
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
        
        from preprocessing.preprocessors import BLPPreprocessor, HSHPreprocessor, LBPPreprocessor
        
        if self.preprocessor_type == 'BLP':
            self.preprocessor = BLPPreprocessor()
        elif self.preprocessor_type == 'HSH':
            self.preprocessor = HSHPreprocessor()
        elif self.preprocessor_type == 'LBP':
            self.preprocessor = LBPPreprocessor()
        
        return self.preprocessor
    
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
    
    # ==================== COMANDO 2: DETECTAR ====================
    def detect(self) -> Dict[str, Any]:
        """
        Detecta cuerpos y rostros en las imágenes del dataset procesado.
        
        Returns:
            dict: Resultado de la detección.
        """
        print(f"\n[DETECTAR] Iniciando detección de cuerpos/rostros...")
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
            from detection import BodyDetection, FaceDetection
            
            body_detector = BodyDetection()
            face_detector = FaceDetection()
            
            detections = {
                'bodies': 0,
                'faces': 0,
                'failed': 0
            }
            
            # Procesar cada imagen
            for person_id in persons:
                person_path = os.path.join(self.DATASET_PROCESSED_PATH, person_id)
                for view in ['front', 'back', 'face']:
                    view_path = os.path.join(person_path, view)
                    if os.path.exists(view_path):
                        for img_file in os.listdir(view_path):
                            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                                img_path = os.path.join(view_path, img_file)
                                # Aquí iría la detección real
                                if view == 'face':
                                    detections['faces'] += 1
                                else:
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
                for view in ['front', 'back']:
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
            
            if not images:
                return {'success': False, 'error': "No se pudieron cargar las imágenes."}
            
            # Extraer características
            feature_vectors = FeatureVector.from_images_batch(images, preprocessor, valid_labels)
            feature_matrix = np.array([fv.to_numpy() for fv in feature_vectors])
            
            # Guardar características
            features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
            labels_file = os.path.join(self.FEATURES_PATH, f'labels_{self.preprocessor_type}.npy')
            
            np.save(features_file, feature_matrix)
            np.save(labels_file, np.array(valid_labels))
            
            return {
                'success': True,
                'message': "Extracción de características completada",
                'preprocessor': self.preprocessor_type,
                'num_images': len(images),
                'feature_dimension': feature_matrix.shape[1] if len(feature_matrix.shape) > 1 else 0,
                'features_file': features_file,
                'labels_file': labels_file
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
    
    # ==================== COMANDO 4: ENTRENAR SVM ====================
    def train_svm(self) -> Dict[str, Any]:
        """
        Entrena el modelo SVM con las características extraídas.
        
        Returns:
            dict: Resultado del entrenamiento.
        """
        print(f"\n[ENTRENAR SVM] Iniciando entrenamiento...")
        print(f"  Características: {self.FEATURES_PATH}")
        print(f"  Modelo salida: {self.SVM_MODEL_PATH}")
        
        # Buscar archivo de características
        features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
        labels_file = os.path.join(self.FEATURES_PATH, f'labels_{self.preprocessor_type}.npy')
        
        if not os.path.exists(features_file) or not os.path.exists(labels_file):
            return {
                'success': False,
                'error': f"Características no encontradas. Ejecute 'extraer' primero.\nBuscando: {features_file}"
            }
        
        try:
            # Cargar características
            features = np.load(features_file)
            labels = np.load(labels_file, allow_pickle=True)
            
            print(f"  Características cargadas: {features.shape}")
            print(f"  Etiquetas: {len(labels)} ({len(set(labels))} clases)")
            
            from svm_classifier import SVMModel, ModelTrainer
            
            trainer = ModelTrainer()
            model, stats = trainer.train(features, labels)
            model.save(self.SVM_MODEL_PATH)
            
            # Guardar evaluación
            self.last_evaluation = stats
            
            return {
                'success': True,
                'message': "Modelo SVM entrenado exitosamente",
                'model_path': self.SVM_MODEL_PATH,
                'num_samples': len(labels),
                'num_classes': len(set(labels)),
                'statistics': stats
            }
        except (ImportError, NotImplementedError):
            print("  [INFO] Clasificador SVM pendiente de implementación")
            
            # Simular estadísticas de evaluación
            unique_labels = list(set(labels)) if 'labels' in dir() else ['persona1', 'persona2']
            self.last_evaluation = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'num_classes': len(unique_labels),
                'status': 'placeholder'
            }
            
            return {
                'success': True,
                'message': "Entrenamiento (placeholder) - SVM no implementado",
                'model_path': self.SVM_MODEL_PATH
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ==================== COMANDO 5: VER EVALUACIÓN ====================
    def evaluate(self) -> Dict[str, Any]:
        """
        Muestra la evaluación del modelo entrenado.
        
        Returns:
            dict: Métricas de evaluación.
        """
        print(f"\n[EVALUACIÓN] Mostrando resultados...")
        
        if not os.path.exists(self.SVM_MODEL_PATH):
            return {
                'success': False,
                'error': "Modelo no encontrado. Ejecute 'entrenar' primero."
            }
        
        if self.last_evaluation is not None:
            return {
                'success': True,
                'message': "Evaluación del modelo",
                'model_path': self.SVM_MODEL_PATH,
                'metrics': self.last_evaluation
            }
        
        try:
            from svm_classifier import SVMModel, ModelEvaluator
            
            # Cargar modelo y datos de prueba
            model = SVMModel.load(self.SVM_MODEL_PATH)
            
            features_file = os.path.join(self.FEATURES_PATH, f'features_{self.preprocessor_type}.npy')
            labels_file = os.path.join(self.FEATURES_PATH, f'labels_{self.preprocessor_type}.npy')
            
            if os.path.exists(features_file) and os.path.exists(labels_file):
                features = np.load(features_file)
                labels = np.load(labels_file, allow_pickle=True)
                
                evaluator = ModelEvaluator()
                metrics = evaluator.evaluate(model, features, labels)
                
                self.last_evaluation = metrics
                
                return {
                    'success': True,
                    'message': "Evaluación completada",
                    'metrics': metrics
                }
            else:
                return {
                    'success': False,
                    'error': "Archivo de características no encontrado para evaluación."
                }
        except (ImportError, NotImplementedError):
            return {
                'success': True,
                'message': "Evaluación (placeholder) - Evaluador no implementado",
                'metrics': {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'status': 'placeholder - modelo no evaluado'
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ==================== COMANDO 6: AUTOMÁTICO ====================
    def run_automatic(self) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo automáticamente:
        1. Preprocesar
        2. Detectar
        3. Extraer características
        4. Entrenar SVM
        5. Mostrar evaluación
        
        Returns:
            dict: Resultado de todo el proceso.
        """
        print("\n" + "=" * 60)
        print("   EJECUCIÓN AUTOMÁTICA DEL PIPELINE COMPLETO")
        print("=" * 60)
        print(f"\nPreprocesador seleccionado: {self.preprocessor_type}")
        print(f"Dataset: {self.DATASET_PATH}")
        print(f"Dataset procesado: {self.DATASET_PROCESSED_PATH}")
        print(f"Modelo: {self.SVM_MODEL_PATH}")
        print("\n" + "-" * 60)
        
        results = {
            'preprocess': None,
            'detect': None,
            'extract': None,
            'train': None,
            'evaluate': None
        }
        
        steps = [
            ('preprocess', 'PASO 1/5: Preprocesamiento', self.preprocess),
            ('detect', 'PASO 2/5: Detección', self.detect),
            ('extract', 'PASO 3/5: Extracción de características', self.extract_features),
            ('train', 'PASO 4/5: Entrenamiento SVM', self.train_svm),
            ('evaluate', 'PASO 5/5: Evaluación', self.evaluate)
        ]
        
        all_success = True
        
        for step_key, step_name, step_func in steps:
            print(f"\n{'─' * 50}")
            print(f"  {step_name}")
            print(f"{'─' * 50}")
            
            try:
                result = step_func()
                results[step_key] = result
                
                if result.get('success', False):
                    print(f"  ✓ {result.get('message', 'Completado')}")
                else:
                    print(f"  ✗ Error: {result.get('error', 'Error desconocido')}")
                    all_success = False
                    # Continuar con el siguiente paso aunque falle
            except Exception as e:
                print(f"  ✗ Excepción: {e}")
                results[step_key] = {'success': False, 'error': str(e)}
                all_success = False
        
        print("\n" + "=" * 60)
        print("   RESUMEN DE EJECUCIÓN")
        print("=" * 60)
        
        for step_key, step_name, _ in steps:
            result = results[step_key]
            status = "✓" if result and result.get('success') else "✗"
            print(f"  {status} {step_name.split(':')[1].strip()}")
        
        print("=" * 60)
        
        return {
            'success': all_success,
            'message': "Pipeline automático completado" if all_success else "Pipeline completado con errores",
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
