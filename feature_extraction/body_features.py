import numpy as np
import cv2
import os


class BodyFeatureExtractor:
    """
    Extractor de características corporales usando HSV.
    
    Este extractor utiliza el preprocesador HSV internamente para
    extraer características de color de las imágenes corporales.
    """
    
    def __init__(self, model='hsv', embedding_dim=None):
        """
        Inicializa el extractor de características corporales.
        
        Args:
            model (str): Modelo a utilizar ('hsv' por defecto).
            embedding_dim (int): Dimensionalidad del embedding (calculada por HSV).
        """
        self.model = model
        
        # Importar y crear instancia del preprocesador HSV
        from preprocessing.preprocessors import HSVPreprocessor
        
        self.preprocessor = HSVPreprocessor()
        self.embedding_dim = self.preprocessor.output_dim
        
        print(f"[BodyFeatureExtractor] Inicializado con HSV")
        print(f"  Dimensión de características: {self.embedding_dim}")
    
    def extract(self, image):
        """
        Extrae características de una imagen corporal.
        
        Args:
            image (np.ndarray): Imagen del cuerpo en formato numpy array.
        
        Returns:
            numpy.ndarray: Vector de características de dimensión embedding_dim.
        """
        # Usar el preprocesador HSV para extraer características
        return self.preprocessor.preprocess(image)
    
    def extract_batch(self, images):
        """
        Extrae características de un lote de imágenes corporales.
        
        Args:
            images (list): Lista de imágenes de cuerpos.
        
        Returns:
            numpy.ndarray: Matriz de dimensión (N, embedding_dim).
        """
        return self.preprocessor.preprocess_batch(images)
    
    def extract_from_directory(self, directory_path):
        """
        Extrae características de todas las imágenes en un directorio.
        
        Args:
            directory_path (str): Ruta del directorio con imágenes.
        
        Returns:
            tuple: (features, file_paths, labels)
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directorio no encontrado: {directory_path}")
        
        # Listar todas las imágenes
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            raise ValueError(f"No se encontraron imágenes en: {directory_path}")
        
        features = []
        file_paths = []
        labels = []
        
        # Determinar la etiqueta basada en el directorio padre
        parent_dir = os.path.basename(directory_path)
        
        print(f"  Procesando {len(image_files)} imágenes de '{directory_path}'...")
        
        for img_file in image_files:
            img_path = os.path.join(directory_path, img_file)
            
            try:
                # Cargar imagen
                img = cv2.imread(img_path)
                if img is None:
                    print(f"    [WARN] No se pudo cargar: {img_file}")
                    continue
                
                # Convertir BGR a RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extraer características
                feature = self.extract(img)
                
                features.append(feature)
                file_paths.append(img_path)
                labels.append(parent_dir)
                
            except Exception as e:
                print(f"    [ERROR] Error procesando {img_file}: {e}")
                continue
        
        if not features:
            raise ValueError(f"No se pudieron extraer características de ninguna imagen")
        
        return np.array(features), file_paths, labels
    
    def extract_front_and_back(self, front_images, back_images):
        """
        Extrae características de vistas frontal y posterior.
        
        Args:
            front_images (list): Lista de imágenes de vista frontal.
            back_images (list): Lista de imágenes de vista posterior.
        
        Returns:
            dict: Diccionario con características separadas.
        """
        front_features = self.extract_batch(front_images)
        back_features = self.extract_batch(back_images)
        
        return {
            'front': front_features,
            'back': back_features
        }
    
    def extract_and_save(self, image_path, output_path):
        """
        Extrae características y las guarda en archivo.
        
        Args:
            image_path (str): Ruta de la imagen.
            output_path (str): Ruta donde guardar las características.
        
        Returns:
            bool: True si se guardó exitosamente.
        """
        try:
            # Cargar imagen
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                return False
            
            # Convertir BGR a RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extraer características
            features = self.extract(img)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Guardar en formato numpy
            np.save(output_path, features)
            
            return True
            
        except Exception as e:
            print(f"Error al extraer y guardar características: {e}")
            return False
    
    def get_info(self):
        """
        Obtiene información del extractor.
        
        Returns:
            dict: Información del extractor.
        """
        return {
            'model': self.model,
            'embedding_dim': self.embedding_dim,
            'preprocessor': self.preprocessor.get_info()
        }