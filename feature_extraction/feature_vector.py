"""
Módulo para gestión de vectores de características - SOLO VIDEOS FRONT.

Implementa operaciones sobre vectores de características como comparación,
distancias, normalización y persistencia.

⚠️ IMPORTANTE: Todos los vectores deben originarse de videos FRONT (frontales).
Se valida el campo metadata['video_type'] = 'front' en cada operación.
"""

import numpy as np
import pickle
import os


class FeatureVector:
    """
    Clase para manejar y gestionar vectores de características.
    
    Attributes:
        vector (numpy.ndarray): Vector de características.
        person_id (str): Identificador de la persona asociada.
        feature_type (str): Tipo de característica ('facial' o 'body').
        metadata (dict): Información adicional del vector.
    """
    
    def __init__(self, vector, person_id=None, feature_type='body', metadata=None, video_type='front'):
        """
        Inicializa un vector de características (SOLO FRONT).
        
        Args:
            vector (numpy.ndarray): Array del vector.
            person_id (str): ID de la persona.
            feature_type (str): Tipo de característica.
            metadata (dict): Información adicional.
            video_type (str): Tipo de video ('front'). Default: 'front'.
                             ❌ No se aceptan videos 'back'.
        
        Raises:
            ValueError: Si video_type no es FRONT.
        """
        # ✓ Validar que sea video FRONT
        if video_type.lower() not in ['front', 'frontal']:
            raise ValueError(
                f"❌ FeatureVector solo acepta videos FRONT. "
                f"Se recibió: {video_type}"
            )
        
        # Validar y convertir a numpy array
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        elif isinstance(vector, np.ndarray):
            vector = vector.astype(np.float32)
        else:
            raise TypeError("El vector debe ser una lista o numpy.ndarray")
        
        # Asegurar que sea un vector 1D
        if vector.ndim != 1:
            vector = vector.flatten()
        
        self.vector = vector
        self.person_id = person_id
        self.feature_type = feature_type
        self.video_type = video_type
        self.metadata = metadata or {}
        
        # Garantizar que metadata marca como FRONT
        self.metadata['video_type'] = 'front'
    
    def compute_distance(self, other_vector, metric='euclidean'):
        """
        Calcula distancia a otro vector.
        
        Args:
            other_vector (FeatureVector o numpy.ndarray): Vector para comparar.
            metric (str): Métrica de distancia ('euclidean', 'cosine', 'manhattan')
        
        Returns:
            float: Distancia entre vectores.
        """
        # Extraer vector si es FeatureVector
        if isinstance(other_vector, FeatureVector):
            other = other_vector.vector
        elif isinstance(other_vector, np.ndarray):
            other = other_vector.flatten().astype(np.float32)
        else:
            raise TypeError("other_vector debe ser FeatureVector o numpy.ndarray")
        
        # Verificar dimensiones
        if len(self.vector) != len(other):
            raise ValueError(f"Los vectores tienen diferentes dimensiones: {len(self.vector)} vs {len(other)}")
        
        if metric == 'euclidean':
            return float(np.linalg.norm(self.vector - other))
        elif metric == 'cosine':
            norm_self = np.linalg.norm(self.vector)
            norm_other = np.linalg.norm(other)
            if norm_self == 0 or norm_other == 0:
                return 1.0  # Máxima distancia si alguno es cero
            cosine_sim = np.dot(self.vector, other) / (norm_self * norm_other)
            return float(1.0 - cosine_sim)  # Convertir similitud a distancia
        elif metric == 'manhattan':
            return float(np.sum(np.abs(self.vector - other)))
        else:
            raise ValueError(f"Métrica no soportada: {metric}. Use 'euclidean', 'cosine' o 'manhattan'")
    
    def compute_similarity(self, other_vector, metric='cosine'):
        """
        Calcula similitud a otro vector.
        
        Args:
            other_vector (FeatureVector o numpy.ndarray): Vector para comparar.
            metric (str): Métrica de similitud ('cosine', 'euclidean')
        
        Returns:
            float: Valor de similitud (0-1 o similar según métrica).
        """
        # Extraer vector si es FeatureVector
        if isinstance(other_vector, FeatureVector):
            other = other_vector.vector
        elif isinstance(other_vector, np.ndarray):
            other = other_vector.flatten().astype(np.float32)
        else:
            raise TypeError("other_vector debe ser FeatureVector o numpy.ndarray")
        
        if len(self.vector) != len(other):
            raise ValueError(f"Los vectores tienen diferentes dimensiones: {len(self.vector)} vs {len(other)}")
        
        if metric == 'cosine':
            norm_self = np.linalg.norm(self.vector)
            norm_other = np.linalg.norm(other)
            if norm_self == 0 or norm_other == 0:
                return 0.0
            return float(np.dot(self.vector, other) / (norm_self * norm_other))
        elif metric == 'euclidean':
            distance = np.linalg.norm(self.vector - other)
            # Convertir distancia a similitud (usando función exponencial)
            return float(np.exp(-distance))
        else:
            raise ValueError(f"Métrica no soportada: {metric}. Use 'cosine' o 'euclidean'")
    
    def normalize(self):
        """
        Normaliza el vector a magnitud unitaria.
        
        Returns:
            FeatureVector: Nueva instancia con vector normalizado.
        """
        magnitude = np.linalg.norm(self.vector)
        if magnitude == 0:
            normalized_vector = self.vector.copy()
        else:
            normalized_vector = self.vector / magnitude
        
        return FeatureVector(
            vector=normalized_vector,
            person_id=self.person_id,
            feature_type=self.feature_type,
            metadata={**self.metadata, 'normalized': True}
        )
    
    def save(self, file_path):
        """
        Guarda el vector en archivo.
        
        Args:
            file_path (str): Ruta donde guardar.
        
        Returns:
            bool: True si se guardó exitosamente.
        """
        try:
            # Crear directorio si no existe
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            data = {
                'vector': self.vector,
                'person_id': self.person_id,
                'feature_type': self.feature_type,
                'metadata': self.metadata
            }
            
            # Determinar formato según extensión
            if file_path.endswith('.npy'):
                np.save(file_path, data, allow_pickle=True)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return True
        except Exception as e:
            print(f"Error al guardar vector: {e}")
            return False
    
    @staticmethod
    def load(file_path):
        """
        Carga un vector desde archivo.
        
        Args:
            file_path (str): Ruta del archivo.
        
        Returns:
            FeatureVector: Instancia cargada.
        """
        try:
            if file_path.endswith('.npy'):
                data = np.load(file_path, allow_pickle=True).item()
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            return FeatureVector(
                vector=data['vector'],
                person_id=data.get('person_id'),
                feature_type=data.get('feature_type', 'body'),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            raise IOError(f"Error al cargar vector desde {file_path}: {e}")
    
    def to_numpy(self):
        """
        Retorna el vector como numpy array.
        
        Returns:
            numpy.ndarray: Copia del vector.
        """
        return self.vector.copy()
    
    @staticmethod
    def from_image(image, extractor):
        """
        Crea un FeatureVector desde una imagen usando un extractor.
        
        Args:
            image (numpy.ndarray): Imagen a procesar.
            extractor: Instancia de extractor (HOG, HSV o LBP).
        
        Returns:
            FeatureVector: Vector de características extraído.
        """
        vector = extractor.extract(image)
        return FeatureVector(
            vector=vector,
            metadata={
                'extractor': extractor.__class__.__name__,
                'output_dim': extractor.output_dim
            }
        )
    
    @staticmethod
    def from_images_batch(images, extractor, person_ids=None):
        """
        Crea múltiples FeatureVectors desde un lote de imágenes.
        
        Args:
            images (list): Lista de imágenes.
            extractor: Instancia de extractor.
            person_ids (list): Lista de IDs de personas (opcional).
        
        Returns:
            list: Lista de FeatureVector.
        """
        vectors = extractor.extract_batch(images)
        result = []
        
        for i, vec in enumerate(vectors):
            person_id = person_ids[i] if person_ids and i < len(person_ids) else None
            result.append(FeatureVector(
                vector=vec,
                person_id=person_id,
                metadata={
                    'extractor': extractor.__class__.__name__,
                    'batch_index': i
                }
            ))
        
        return result
    
    def __len__(self):
        """Retorna la dimensionalidad del vector."""
        return len(self.vector)
    
    def __repr__(self):
        """Representación en string del vector."""
        return (f"FeatureVector(dim={len(self.vector)}, "
                f"person_id='{self.person_id}', "
                f"type='{self.feature_type}')")
    
    def __eq__(self, other):
        """Compara dos FeatureVectors."""
        if not isinstance(other, FeatureVector):
            return False
        return np.allclose(self.vector, other.vector)