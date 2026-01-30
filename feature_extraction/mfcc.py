"""
Módulo para extracción de características MFCC (Mel-Frequency Cepstral Coefficients).

Los coeficientes MFCC se utilizan principalmente para procesamiento de audio y voz,
capturando características auditivas perceptualmente relevantes.
"""

import numpy as np
import librosa
import librosa.feature
from typing import Optional, Union


class MFCCExtractor:
    """
    Extractor de características MFCC (Mel-Frequency Cepstral Coefficients).
    
    Diseñado para extraer características de audio que pueden estar presentes
    en datos de video/audio relacionados con detección de cuerpos (ej: voz, movimiento).
    
    Attributes:
        n_mfcc (int): Número de coeficientes MFCC a extraer.
        n_fft (int): Tamaño de la ventana FFT.
        hop_length (int): Número de muestras entre sucesivos frames.
        sr (int): Frecuencia de muestreo (Hz).
    """
    
    def __init__(
        self, 
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        sr: int = 22050
    ):
        """
        Inicializa el extractor MFCC.
        
        Args:
            n_mfcc (int): Número de coeficientes MFCC. Default: 13.
            n_fft (int): Tamaño FFT. Default: 2048.
            hop_length (int): Salto entre frames. Default: 512.
            sr (int): Frecuencia de muestreo en Hz. Default: 22050.
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
    
    def extract(self, audio_data: Union[np.ndarray, str]) -> np.ndarray:
        """
        Extrae características MFCC de una señal de audio.
        
        Args:
            audio_data (np.ndarray or str): 
                - Array de audio (1D o 2D con muestras de audio)
                - Ruta a archivo de audio (str)
        
        Returns:
            np.ndarray: Vector de características MFCC aplanado.
                       Dimensión: (n_mfcc * n_frames,)
        
        Raises:
            TypeError: Si audio_data no es np.ndarray o str.
            ValueError: Si audio_data está vacío.
        """
        # Cargar audio si es ruta
        if isinstance(audio_data, str):
            try:
                audio_data, sr = librosa.load(audio_data, sr=self.sr)
            except Exception as e:
                raise ValueError(f"No se pudo cargar el archivo de audio: {e}")
        
        elif isinstance(audio_data, np.ndarray):
            # Si es un array, asumimos que está en formato de audio
            pass
        else:
            raise TypeError(
                f"Se espera np.ndarray o str (ruta), se recibió {type(audio_data)}"
            )
        
        # Validaciones
        if audio_data.size == 0:
            raise ValueError("Los datos de audio están vacíos")
        
        # Asegurar que sea 1D
        if audio_data.ndim > 1:
            # Si tiene múltiples canales, promediar
            audio_data = np.mean(audio_data, axis=0)
        
        # Extraer MFCC
        try:
            mfcc_features = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Aplanar el resultado: (n_mfcc, n_frames) -> (n_mfcc * n_frames,)
            return mfcc_features.flatten().astype(np.float32)
        
        except Exception as e:
            raise ValueError(f"Error extrayendo MFCC: {e}")
    
    def extract_batch(self, audio_list: list) -> np.ndarray:
        """
        Extrae características MFCC de un lote de audios.
        
        Args:
            audio_list (list): Lista de datos de audio (arrays o rutas).
        
        Returns:
            np.ndarray: Matriz de características con padding.
                       Dimensión: (N, max_features)
        
        Raises:
            ValueError: Si la lista está vacía.
        """
        if not isinstance(audio_list, list) or len(audio_list) == 0:
            raise ValueError("Se espera una lista no vacía de audios")
        
        features_list = []
        max_length = 0
        
        # Primera pasada: extraer características y encontrar max_length
        for audio_data in audio_list:
            try:
                features = self.extract(audio_data)
                features_list.append(features)
                max_length = max(max_length, len(features))
            except Exception as e:
                print(f"Advertencia: error al procesar audio: {e}. Omitiendo...")
                # Usar vector de ceros como fallback
                features_list.append(None)
        
        # Segunda pasada: padding
        padded_features = []
        for features in features_list:
            if features is not None:
                # Padding: rellenar con ceros al final
                padded = np.pad(
                    features,
                    (0, max_length - len(features)),
                    mode='constant',
                    constant_values=0
                )
            else:
                # Si hubo error, usar vector de ceros
                padded = np.zeros(max_length, dtype=np.float32)
            
            padded_features.append(padded)
        
        return np.array(padded_features, dtype=np.float32)
    
    def extract_statistics(self, audio_data: Union[np.ndarray, str]) -> np.ndarray:
        """
        Extrae características MFCC y calcula estadísticas sobre los frames.
        
        Útil para obtener un vector de dimensión fija independiente del audio.
        
        Args:
            audio_data (np.ndarray or str): Datos de audio o ruta.
        
        Returns:
            np.ndarray: Vector de características combinadas.
                       [mean_1, std_1, ..., mean_n, std_n] para cada coeficiente
                       Dimensión: (2 * n_mfcc,)
        """
        # Cargar audio si es ruta
        if isinstance(audio_data, str):
            try:
                audio_data, sr = librosa.load(audio_data, sr=self.sr)
            except Exception as e:
                raise ValueError(f"No se pudo cargar el archivo de audio: {e}")
        
        elif isinstance(audio_data, np.ndarray):
            pass
        else:
            raise TypeError(f"Se espera np.ndarray o str, se recibió {type(audio_data)}")
        
        if audio_data.size == 0:
            raise ValueError("Los datos de audio están vacíos")
        
        # Asegurar 1D
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        # Extraer MFCC: (n_mfcc, n_frames)
        try:
            mfcc_features = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Calcular estadísticas por coeficiente
            means = np.mean(mfcc_features, axis=1)  # (n_mfcc,)
            stds = np.std(mfcc_features, axis=1)    # (n_mfcc,)
            
            # Combinar: [mean_1, std_1, mean_2, std_2, ...]
            stats = np.concatenate([means, stds]).astype(np.float32)
            return stats
        
        except Exception as e:
            raise ValueError(f"Error extrayendo estadísticas MFCC: {e}")
    
    def get_feature_names(self, mode: str = 'flat') -> list:
        """
        Retorna los nombres de las características extraídas.
        
        Args:
            mode (str): 'flat' para aplanado, 'stats' para estadísticas.
        
        Returns:
            list: Lista de nombres de características.
        """
        if mode == 'flat':
            # Aproximado, basado en número típico de frames
            n_frames_approx = 100  # Valor aproximado
            names = [f"mfcc_coeff_{i}_{j}" 
                    for i in range(self.n_mfcc) 
                    for j in range(n_frames_approx)]
            return names
        
        elif mode == 'stats':
            names = []
            for i in range(self.n_mfcc):
                names.append(f"mfcc_mean_{i}")
            for i in range(self.n_mfcc):
                names.append(f"mfcc_std_{i}")
            return names
        
        else:
            raise ValueError(f"Modo desconocido: {mode}")
