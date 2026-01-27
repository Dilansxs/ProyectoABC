import os
import shutil
from .frame_extraction import FrameExtraction
from .data_augmentation import DataAugmentation


class PreprocessingPipeline:
    """
    Pipeline completo: extracción de frames + detección YOLOv8 + augmentation.
    """
    
    def __init__(self, dataset_path, output_path, fps=10, use_yolo=True):
        """
        Inicializa el pipeline de preprocesamiento.
        
        Args:
            dataset_path (str): Ruta del dataset original.
            output_path (str): Ruta del dataset procesado.
            fps (int): Fotogramas por segundo a extraer.
            use_yolo (bool): Si es True, usa YOLOv8 para detectar cuerpos.
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.fps = fps
        self.use_yolo = use_yolo
        
        os.makedirs(self.output_path, exist_ok=True)
        
        self.temp_frames_path = os.path.join(output_path, '_temp_frames')
        os.makedirs(self.temp_frames_path, exist_ok=True)
        
        self.frame_extractor = FrameExtraction(fps=fps, output_path=self.temp_frames_path)
        self.data_augmenter = DataAugmentation()
        
        # Inicializar detector de cuerpos si se usa YOLO
        if self.use_yolo:
            try:
                from detection import BodyDetection
                self.body_detector = BodyDetection(confidence_threshold=0.3)
                print("[Pipeline] YOLOv8 habilitado para detección de cuerpos")
            except Exception as e:
                print(f"[WARN] No se pudo cargar YOLOv8: {e}")
                print("[INFO] Continuando sin detección automática")
                self.use_yolo = False
                self.body_detector = None
        else:
            self.body_detector = None
        
        self.results = {}
    
    def run_full_pipeline(self, augmentation_multiplier=3):
        """
        Ejecuta el pipeline completo de preprocesamiento.
        """
        print("\n" + "="*60)
        print("   PIPELINE COMPLETO DE PREPROCESAMIENTO")
        if self.use_yolo:
            print("   (con detección YOLOv8)")
        print("="*60)
        
        # PASO 1: Extracción de fotogramas
        print("\n[PASO 1/4] Extrayendo fotogramas de videos...")
        extraction_stats = self.frame_extractor.process_dataset(self.dataset_path)
        self.results['extraction'] = extraction_stats
        
        print(f"\n✓ Extracción completada:")
        print(f"  - Videos procesados: {extraction_stats['total_videos']}")
        print(f"  - Frames extraídos: {extraction_stats['total_frames']}")
        print(f"  - Personas: {extraction_stats['total_persons']}")
        
        # PASO 2: Detección de cuerpos con YOLO (opcional)
        if self.use_yolo and self.body_detector is not None:
            print("\n[PASO 2/4] Detectando cuerpos con YOLOv8...")
            detection_stats = self._detect_bodies()
            self.results['detection'] = detection_stats
            
            print(f"\n✓ Detección completada:")
            print(f"  - Imágenes procesadas: {detection_stats.get('images_processed', 0)}")
            print(f"  - Cuerpos detectados: {detection_stats.get('bodies_detected', 0)}")
        else:
            print("\n[PASO 2/4] Omitiendo detección (YOLOv8 no disponible)")
            self.results['detection'] = {'skipped': True}
        
        # PASO 3: Data augmentation
        print("\n[PASO 3/4] Aplicando data augmentation...")
        
        # Decidir qué carpeta usar como entrada
        if self.use_yolo and self.results.get('detection', {}).get('bodies_detected', 0) > 0:
            augmentation_input = self.temp_frames_path + '_detected'
        else:
            augmentation_input = self.temp_frames_path
        
        augmentation_stats = self.data_augmenter.augment_dataset(
            augmentation_input,
            self.output_path,
            multiplier=augmentation_multiplier
        )
        self.results['augmentation'] = augmentation_stats
        
        print(f"\n✓ Augmentation completada:")
        print(f"  - Imágenes originales: {augmentation_stats['original_images']}")
        print(f"  - Imágenes aumentadas: {augmentation_stats['augmented_images']}")
        print(f"  - Total final: {augmentation_stats['total_images']}")
        
        # PASO 4: Limpieza
        print("\n[PASO 4/4] Limpiando archivos temporales...")
        self._cleanup_temp()
        
        print("\n" + "="*60)
        print("   PREPROCESAMIENTO COMPLETADO")
        print("="*60)
        print(f"\nDataset procesado guardado en:")
        print(f"  {self.output_path}")
        print("\nPróximos pasos:")
        print("  1. Extraer características (HSV)")
        print("  2. Entrenar modelo SVM")
        print("  3. Evaluar desempeño")
        print("="*60 + "\n")
        
        return {
            'extraction': extraction_stats,
            'detection': self.results.get('detection', {}),
            'augmentation': augmentation_stats,
            'output_path': self.output_path
        }
    
    def _detect_bodies(self):
        """
        Detecta cuerpos en los frames extraídos usando YOLOv8.
        """
        import cv2
        
        stats = {
            'images_processed': 0,
            'bodies_detected': 0,
            'bodies_saved': 0,
            'persons': {}
        }
        
        # Crear carpeta para frames con cuerpos detectados
        detected_path = self.temp_frames_path + '_detected'
        os.makedirs(detected_path, exist_ok=True)
        
        # Procesar cada persona
        for person_name in os.listdir(self.temp_frames_path):
            person_path = os.path.join(self.temp_frames_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            print(f"\n  Procesando: {person_name}")
            stats['persons'][person_name] = {
                'images': 0,
                'bodies': 0
            }
            
            # Procesar cada vista
            for view_type in ['front', 'back']:
                view_path = os.path.join(person_path, view_type)
                
                if not os.path.exists(view_path):
                    continue
                
                # Crear carpeta de salida
                output_view_path = os.path.join(detected_path, person_name, view_type)
                os.makedirs(output_view_path, exist_ok=True)
                
                # Procesar imágenes
                image_files = [f for f in os.listdir(view_path) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                for img_file in image_files:
                    img_path = os.path.join(view_path, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    stats['images_processed'] += 1
                    stats['persons'][person_name]['images'] += 1
                    
                    # Detectar cuerpos
                    bodies = self.body_detector.detect_and_crop(img)
                    
                    # Guardar cada cuerpo detectado
                    for i, body in enumerate(bodies):
                        base_name = os.path.splitext(img_file)[0]
                        output_file = f"{base_name}_body{i}.jpg"
                        output_path = os.path.join(output_view_path, output_file)
                        
                        cv2.imwrite(output_path, body)
                        
                        stats['bodies_detected'] += 1
                        stats['bodies_saved'] += 1
                        stats['persons'][person_name]['bodies'] += 1
                
                print(f"    {view_type}: {stats['persons'][person_name]['bodies']} cuerpos detectados")
        
        return stats
    
    def _cleanup_temp(self):
        """
        Limpia las carpetas temporales.
        """
        temp_folders = [
            self.temp_frames_path,
            self.temp_frames_path + '_detected'
        ]
        
        for folder in temp_folders:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"  ✓ Eliminada carpeta temporal: {os.path.basename(folder)}")
                except Exception as e:
                    print(f"  ⚠ No se pudo eliminar {folder}: {e}")
    
    def extract_frames_only(self):
        """
        Ejecuta solo la etapa de extracción de fotogramas.
        """
        print("\n[EXTRACCIÓN DE FOTOGRAMAS]")
        stats = self.frame_extractor.process_dataset(self.dataset_path)
        self.results['extraction'] = stats
        return stats
    
    def get_pipeline_status(self):
        """
        Obtiene el estado actual del pipeline.
        """
        status = {
            'dataset_path': self.dataset_path,
            'output_path': self.output_path,
            'temp_path': self.temp_frames_path,
            'fps': self.fps,
            'use_yolo': self.use_yolo,
            'results': self.results
        }
        
        if os.path.exists(self.output_path):
            total_files = 0
            for root, dirs, files in os.walk(self.output_path):
                total_files += len([f for f in files 
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            status['output_files'] = total_files
        
        return status
    
    def clean_output(self, force=False):
        """
        Limpia los archivos de salida.
        """
        if not force:
            confirm = input(f"¿Eliminar todo el contenido de {self.output_path}? (s/n): ")
            if confirm.lower() != 's':
                print("Operación cancelada.")
                return False
        
        try:
            if os.path.exists(self.output_path):
                shutil.rmtree(self.output_path)
                os.makedirs(self.output_path, exist_ok=True)
                print(f"✓ Directorio limpiado: {self.output_path}")
                return True
        except Exception as e:
            print(f"✗ Error al limpiar: {e}")
            return False