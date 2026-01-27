import numpy as np
import cv2


class HSVPreprocessor:
    """
    Preprocesador basado en histogramas HSV.
    
    Extrae características de color mediante histogramas 3D del espacio HSV,
    dividiendo la imagen en regiones espaciales para capturar información local.
    
    Attributes:
        name (str): Nombre del preprocesador.
        h_bins (int): Número de bins para el canal Hue.
        s_bins (int): Número de bins para el canal Saturation.
        v_bins (int): Número de bins para el canal Value.
        grid_size (tuple): División espacial de la imagen (filas, columnas).
        output_dim (int): Dimensionalidad del vector de salida.
    """
    
    def __init__(self, h_bins=16, s_bins=16, v_bins=16, grid_size=(4, 4)):
        """
        Inicializa el preprocesador HSV.
        
        Args:
            h_bins (int): Bins para Hue (0-180). Default: 16.
            s_bins (int): Bins para Saturation (0-255). Default: 16.
            v_bins (int): Bins para Value (0-255). Default: 16.
            grid_size (tuple): División espacial (filas, columnas). Default: (4, 4).
        """
        self.name = 'HSV'
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.grid_size = grid_size
        
        # Dimensión del histograma por celda: h_bins * s_bins * v_bins
        hist_per_cell = h_bins * s_bins * v_bins
        # Dimensión total: histograma por celda * número de celdas
        self.output_dim = hist_per_cell * grid_size[0] * grid_size[1]
    
    def preprocess(self, image):
        """
        Preprocesa una imagen extrayendo características HSV.
        
        Args:
            image (np.ndarray): Imagen en formato RGB o BGR (H, W, C).
        
        Returns:
            np.ndarray: Vector de características de dimensión (output_dim,).
        """
        # Redimensionar a tamaño estándar para cuerpos
        image = cv2.resize(image, (128, 256))  # Ancho x Alto
        
        # Convertir a HSV
        if len(image.shape) == 2:
            # Si es escala de grises, convertir a RGB primero
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convertir RGB a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calcular histogramas espaciales
        feature_vector = self._compute_spatial_histograms(hsv)
        
        # Normalizar el vector
        feature_vector = feature_vector / (np.sum(feature_vector) + 1e-7)
        
        return feature_vector.astype(np.float32)
    
    def preprocess_batch(self, images):
        """
        Preprocesa un lote de imágenes.
        
        Args:
            images (list): Lista de imágenes en formato numpy array.
        
        Returns:
            np.ndarray: Matriz de características de dimensión (N, output_dim).
        """
        features = []
        for img in images:
            features.append(self.preprocess(img))
        return np.array(features)
    
    def _compute_spatial_histograms(self, hsv_image):
        """
        Calcula histogramas HSV para cada región espacial de la imagen.
        
        Args:
            hsv_image (np.ndarray): Imagen en espacio HSV.
        
        Returns:
            np.ndarray: Vector concatenado de histogramas.
        """
        height, width = hsv_image.shape[:2]
        grid_h, grid_w = self.grid_size
        
        # Tamaño de cada celda
        cell_h = height // grid_h
        cell_w = width // grid_w
        
        histograms = []
        
        # Iterar sobre cada celda de la grilla
        for i in range(grid_h):
            for j in range(grid_w):
                # Calcular límites de la celda
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < grid_h - 1 else height
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < grid_w - 1 else width
                
                # Extraer la región de la celda
                cell = hsv_image[y_start:y_end, x_start:x_end]
                
                # Calcular histograma 3D HSV para esta celda
                hist = self._compute_hsv_histogram(cell)
                histograms.append(hist)
        
        # Concatenar todos los histogramas
        feature_vector = np.concatenate(histograms)
        
        return feature_vector
    
    def _compute_hsv_histogram(self, hsv_cell):
        """
        Calcula histograma 3D del espacio HSV para una región.
        
        Args:
            hsv_cell (np.ndarray): Región de la imagen en espacio HSV.
        
        Returns:
            np.ndarray: Histograma 3D aplanado.
        """
        # Separar canales
        h_channel = hsv_cell[:, :, 0]  # Hue: 0-180
        s_channel = hsv_cell[:, :, 1]  # Saturation: 0-255
        v_channel = hsv_cell[:, :, 2]  # Value: 0-255
        
        # Calcular histograma 3D
        hist, _ = np.histogramdd(
            np.column_stack((
                h_channel.ravel(),
                s_channel.ravel(),
                v_channel.ravel()
            )),
            bins=(self.h_bins, self.s_bins, self.v_bins),
            range=((0, 180), (0, 256), (0, 256))
        )
        
        # Aplanar el histograma 3D
        hist_flat = hist.flatten().astype(np.float32)
        
        return hist_flat
    
    def get_info(self):
        """
        Obtiene información del preprocesador.
        
        Returns:
            dict: Diccionario con información del preprocesador.
        """
        return {
            'name': self.name,
            'output_dim': self.output_dim,
            'h_bins': self.h_bins,
            's_bins': self.s_bins,
            'v_bins': self.v_bins,
            'grid_size': self.grid_size,
            'type': self.__class__.__name__
        }
    
    def __repr__(self):
        return (f"HSVPreprocessor(h_bins={self.h_bins}, s_bins={self.s_bins}, "
                f"v_bins={self.v_bins}, grid_size={self.grid_size}, "
                f"output_dim={self.output_dim})")