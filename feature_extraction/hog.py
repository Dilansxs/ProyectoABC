import numpy as np
import cv2


class HOGExtractor:
    """
    Extractor HOG (Histogram of Oriented Gradients).
    
    Extrae características basadas en la distribución de orientaciones de gradientes
    en diferentes regiones de la imagen.
    
    Attributes:
        orientations (int): Número de bins para el histograma de orientaciones.
        pixels_per_cell (tuple): Tamaño de las celdas (alto, ancho) en píxeles.
        cells_per_block (tuple): Número de celdas por bloque (alto, ancho).
        output_dim (int): Dimensionalidad del vector de salida.
    """
    
    def __init__(self, orientations: int = 9, pixels_per_cell: tuple = (8, 8),
                 cells_per_block: tuple = (2, 2), output_dim: int = 1764):
        """
        Inicializa el extractor HOG.
        
        Args:
            orientations (int): Número de bins de orientación. Default: 9.
            pixels_per_cell (tuple): Píxeles por celda (alto, ancho). Default: (8, 8).
            cells_per_block (tuple): Celdas por bloque (alto, ancho). Default: (2, 2).
            output_dim (int): Dimensionalidad del vector de salida. Default: 1764.
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.output_dim = output_dim
        
        # Validar parámetros
        if orientations < 1:
            raise ValueError("orientations debe ser al menos 1")
        if pixels_per_cell[0] < 1 or pixels_per_cell[1] < 1:
            raise ValueError("pixels_per_cell debe tener valores positivos")
        if cells_per_block[0] < 1 or cells_per_block[1] < 1:
            raise ValueError("cells_per_block debe tener valores positivos")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae características HOG de una imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada (H, W, C) o (H, W).
        
        Returns:
            np.ndarray: Vector de características HOG de dimensión (output_dim,).
        """
        # Validar formato de imagen
        if not isinstance(image, np.ndarray):
            raise TypeError("La imagen debe ser un numpy.ndarray")
        
        if image.size == 0:
            raise ValueError("La imagen está vacía")
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calcular gradientes
        magnitude, orientation = self._compute_gradients(gray)
        
        # Construir histogramas por celda
        histograms = self._build_histograms(magnitude, orientation)
        
        # Normalizar bloques
        normalized = self._normalize_blocks(histograms)
        
        # Concatenar vector final
        feature_vector = self._flatten_features(normalized)
        
        # Redimensionar a output_dim
        if len(feature_vector) > self.output_dim:
            feature_vector = feature_vector[:self.output_dim]
        elif len(feature_vector) < self.output_dim:
            # Rellenar con ceros si es necesario
            padding = np.zeros(self.output_dim - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
        
        return feature_vector.astype(np.float32)
    
    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extrae características HOG de un lote de imágenes.
        
        Args:
            images (list): Lista de imágenes (numpy arrays).
        
        Returns:
            np.ndarray: Matriz de características de dimensión (N, output_dim).
        """
        if not isinstance(images, (list, np.ndarray)):
            raise TypeError("images debe ser una lista o numpy.ndarray")
        
        if len(images) == 0:
            raise ValueError("La lista de imágenes está vacía")
        
        # Extraer características de cada imagen
        features = []
        for img in images:
            feat = self.extract(img)
            features.append(feat)
        
        # Apilar resultados en una matriz
        return np.array(features, dtype=np.float32)
    
    def _compute_gradients(self, image: np.ndarray) -> tuple:
        """
        Calcula los gradientes (magnitud y orientación) de una imagen.
        
        Args:
            image (np.ndarray): Imagen en escala de grises.
        
        Returns:
            tuple: (magnitud, orientación) como numpy arrays.
        """
        # Convertir a float para precisión
        image = image.astype(np.float32)
        
        # Aplicar filtro Sobel en X e Y
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        
        # Calcular magnitud: sqrt(Gx^2 + Gy^2)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Calcular orientación: atan2(Gy, Gx)
        orientation = np.arctan2(gy, gx) * (180 / np.pi)
        
        # Convertir orientación a rango [0, 180)
        orientation = orientation % 180
        
        return magnitude, orientation
    
    def _build_histograms(self, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Construye histogramas de orientaciones por celda.
        
        Args:
            magnitude (np.ndarray): Magnitud de gradientes.
            orientation (np.ndarray): Orientación de gradientes.
        
        Returns:
            np.ndarray: Matriz de histogramas (n_cells_y, n_cells_x, orientations).
        """
        h, w = magnitude.shape
        cell_h, cell_w = self.pixels_per_cell
        
        # Calcular número de celdas
        n_cells_y = h // cell_h
        n_cells_x = w // cell_w
        
        # Inicializar matriz de histogramas
        histograms = np.zeros((n_cells_y, n_cells_x, self.orientations))
        
        # Tamaño de cada bin de orientación
        bin_size = 180.0 / self.orientations
        
        # Para cada celda, construir histograma
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Extraer región de la celda
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation[y_start:y_end, x_start:x_end]
                
                # Construir histograma usando magnitud como peso
                for y in range(cell_mag.shape[0]):
                    for x in range(cell_mag.shape[1]):
                        angle = cell_ori[y, x]
                        mag = cell_mag[y, x]
                        
                        # Determinar bin
                        bin_idx = int(angle / bin_size) % self.orientations
                        
                        # Acumular magnitud en el bin
                        histograms[i, j, bin_idx] += mag
        
        return histograms
    
    def _normalize_blocks(self, histograms: np.ndarray) -> np.ndarray:
        """
        Normaliza bloques de histogramas.
        
        Args:
            histograms (np.ndarray): Matriz de histogramas por celda.
        
        Returns:
            np.ndarray: Histogramas normalizados.
        """
        n_cells_y, n_cells_x, _ = histograms.shape
        block_h, block_w = self.cells_per_block
        
        # Calcular número de bloques
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1
        
        if n_blocks_y <= 0 or n_blocks_x <= 0:
            # Si no hay suficientes celdas, retornar histogramas sin normalizar
            return histograms
        
        normalized_blocks = []
        
        # Para cada bloque
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                # Extraer bloque de histogramas
                block = histograms[i:i+block_h, j:j+block_w, :]
                
                # Concatenar histogramas del bloque
                block_vector = block.flatten()
                
                # Normalizar con L2 norm
                norm = np.linalg.norm(block_vector)
                if norm > 0:
                    block_vector = block_vector / (norm + 1e-6)
                
                normalized_blocks.append(block_vector)
        
        return np.array(normalized_blocks)
    
    def _flatten_features(self, normalized_histograms: np.ndarray) -> np.ndarray:
        """
        Aplana los histogramas normalizados en un vector 1D.
        
        Args:
            normalized_histograms (np.ndarray): Histogramas normalizados por bloque.
        
        Returns:
            np.ndarray: Vector 1D de características.
        """
        # Concatenar todos los bloques normalizados
        return normalized_histograms.flatten()