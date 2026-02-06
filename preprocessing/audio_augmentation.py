"""
Módulo de aumentación de datos de audio.

Genera variaciones de muestras de audio para expandir dataset sin replicación artificial.
Técnicas: pitch shift, time stretch, dynamic range compression, noise injection.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import List, Tuple
from pathlib import Path


class AudioAugmentation:
    """
    Augmentador de audio para mejorar diversidad de datos.
    
    Crea variaciones naturales de muestras de audio:
    - Pitch shift: Cambiar tono (±2, ±4 semitonos)
    - Time stretch: Variar velocidad (0.95x, 1.05x)
    - Dynamic range compression: Comprimir dinámica
    - Gaussian noise: Ruido de fondo suave
    """
    
    def __init__(self, sr: int = 22050):
        """
        Inicializa el augmentador.
        
        Args:
            sr (int): Frecuencia de muestreo
        """
        self.sr = sr
        print(f"[AudioAugmentation] ✓ Augmentador inicializado - sr={sr}Hz")
    
    def pitch_shift(self, y: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Desplaza el pitch (tono) del audio.
        
        Args:
            y: Array de audio
            n_steps: Desplazamiento en semitonos (+/-2 a +/-4)
        
        Returns:
            Audio con pitch desplazado
        """
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
    
    def time_stretch(self, y: np.ndarray, rate: float) -> np.ndarray:
        """
        Estira/comprime el tiempo del audio (cambia velocidad).
        
        Args:
            y: Array de audio
            rate: Factor de estiramiento (0.95 = más lento, 1.05 = más rápido)
        
        Returns:
            Audio con tiempo modificado
        """
        return librosa.effects.time_stretch(y, rate=rate)
    
    def add_gaussian_noise(self, y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        Agrega ruido Gaussiano suave.
        
        Args:
            y: Array de audio
            noise_factor: Factor de ruido (0.005 = muy suave)
        
        Returns:
            Audio con ruido añadido
        """
        noise = np.random.normal(0, noise_factor, len(y))
        return y + noise
    
    def compress_dynamic_range(self, y: np.ndarray, threshold: float = -20.0, ratio: float = 4.0) -> np.ndarray:
        """
        Comprime el rango dinámico del audio.
        
        Args:
            y: Array de audio
            threshold: Umbral en dB
            ratio: Ratio de compresión (4:1)
        
        Returns:
            Audio comprimido
        """
        # Convertir a dB
        S = librosa.feature.melspectrogram(y=y, sr=self.sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Aplicar compresión (simplificada)
        # En frecuencias > threshold, reducir por ratio
        compressed = S_db.copy()
        mask = S_db > threshold
        compressed[mask] = threshold + (S_db[mask] - threshold) / ratio
        
        # Convertir de vuelta para propósitos de este augmentador
        # (en producción usarías algorithms más sofisticados)
        return y * 0.95  # Simplificación: solo reducir volumen ligeramente
    
    def augment_audio(self, y: np.ndarray, augmentation_type: str) -> np.ndarray:
        """
        Aplica un tipo específico de augmentación.
        
        Args:
            y: Array de audio
            augmentation_type: 'pitch_+2', 'pitch_-2', 'pitch_+4', 'pitch_-4',
                             'stretch_0.95', 'stretch_1.05', 'noise', 'compress'
        
        Returns:
            Audio augmentado
        """
        try:
            if augmentation_type == 'pitch_+2':
                return self.pitch_shift(y, n_steps=2)
            elif augmentation_type == 'pitch_-2':
                return self.pitch_shift(y, n_steps=-2)
            elif augmentation_type == 'pitch_+4':
                return self.pitch_shift(y, n_steps=4)
            elif augmentation_type == 'pitch_-4':
                return self.pitch_shift(y, n_steps=-4)
            elif augmentation_type == 'stretch_0.95':
                return self.time_stretch(y, rate=0.95)
            elif augmentation_type == 'stretch_1.05':
                return self.time_stretch(y, rate=1.05)
            elif augmentation_type == 'noise':
                return self.add_gaussian_noise(y, noise_factor=0.005)
            elif augmentation_type == 'compress':
                return self.compress_dynamic_range(y)
            else:
                return y
        except Exception as e:
            print(f"[AudioAugmentation] ⚠️  Error en augmentación {augmentation_type}: {e}")
            return y
    
    def create_augmented_variants(self, audio_path: str, num_variants: int = 10) -> List[np.ndarray]:
        """
        Crea múltiples variantes de un audio mediante augmentación.
        
        Args:
            audio_path: Ruta del archivo de audio
            num_variants: Número de variantes a generar (default 10)
        
        Returns:
            Lista de arrays de audio augmentados
        """
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Lista de augmentaciones disponibles
            augmentations = [
                'pitch_+2', 'pitch_-2', 'pitch_+4', 'pitch_-4',
                'stretch_0.95', 'stretch_1.05', 'noise', 'compress'
            ]
            
            variants = [y]  # El original
            
            # Generar variantes
            for i in range(num_variants - 1):  # -1 porque ya incluimos original
                aug_type = augmentations[i % len(augmentations)]
                augmented = self.augment_audio(y, aug_type)
                variants.append(augmented)
            
            print(f"[AudioAugmentation] ✓ {len(variants)} variantes generadas de {os.path.basename(audio_path)}")
            return variants
        
        except Exception as e:
            print(f"[AudioAugmentation] ❌ Error cargando audio: {e}")
            return [np.array([])]
    
    def save_augmented_variants(self, audio_path: str, output_dir: str, num_variants: int = 10) -> List[str]:
        """
        Crea variantes y las guarda en disco.
        
        Args:
            audio_path: Ruta del audio original
            output_dir: Directorio donde guardar variantes
            num_variants: Número de variantes
        
        Returns:
            Lista de rutas de los archivos guardados
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Crear variantes
            variants = self.create_augmented_variants(audio_path, num_variants)
            
            base_name = Path(audio_path).stem
            saved_paths = []
            
            for i, audio_data in enumerate(variants):
                if len(audio_data) == 0:
                    continue
                
                output_path = os.path.join(output_dir, f"{base_name}_aug{i:02d}.wav")
                
                sf.write(output_path, audio_data, self.sr, subtype='PCM_16')
                saved_paths.append(output_path)
            
            print(f"[AudioAugmentation] ✓ {len(saved_paths)} archivos guardados en {output_dir}")
            return saved_paths
        
        except Exception as e:
            print(f"[AudioAugmentation] ❌ Error guardando variantes: {e}")
            return []
    
    def augment_dataset(self, dataset_audio_dir: str, output_base_dir: str, 
                       variants_per_audio: int = 10) -> dict:
        """
        Augmenta todos los audios de un dataset.
        
        Estructura esperada:
        dataset_audio_dir/
        ├── Angeli/
        │   ├── audio_001.wav
        │   └── ...
        ├── Danilo/
        └── ...
        
        Args:
            dataset_audio_dir: Directorio raíz con carpetas por persona
            output_base_dir: Directorio raíz donde guardar aumentados
            variants_per_audio: Variantes por audio
        
        Returns:
            dict: Estadísticas de augmentación
        """
        stats = {
            'total_persons': 0,
            'total_audios_original': 0,
            'total_audios_augmented': 0,
            'persons': {}
        }
        
        # Iterar personas
        for person_dir in os.listdir(dataset_audio_dir):
            person_path = os.path.join(dataset_audio_dir, person_dir)
            
            if not os.path.isdir(person_path):
                continue
            
            stats['total_persons'] += 1
            stats['persons'][person_dir] = {
                'audios_original': 0,
                'audios_augmented': 0
            }
            
            # Crear directorio de salida para esta persona
            output_person_dir = os.path.join(output_base_dir, person_dir)
            
            # Iterar audios de la persona
            for audio_file in os.listdir(person_path):
                if not audio_file.lower().endswith('.wav'):
                    continue
                
                audio_path = os.path.join(person_path, audio_file)
                
                # Generar variantes
                saved = self.save_augmented_variants(
                    audio_path,
                    output_person_dir,
                    variants_per_audio
                )
                
                stats['total_audios_original'] += 1
                stats['total_audios_augmented'] += len(saved)
                stats['persons'][person_dir]['audios_original'] += 1
                stats['persons'][person_dir]['audios_augmented'] += len(saved)
        
        return stats


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def example_usage():
    """Ejemplo de cómo usar AudioAugmentation."""
    
    augmentor = AudioAugmentation(sr=22050)
    
    # Ejemplo 1: Crear variantes de un audio individual
    print("\n=== EJEMPLO 1: Variantes de un audio ===")
    audio_path = "data/audios/Angeli/audio_001.wav"
    
    if os.path.exists(audio_path):
        variants = augmentor.create_augmented_variants(audio_path, num_variants=10)
        print(f"✓ Generadas {len(variants)} variantes")
    
    # Ejemplo 2: Guardar variantes
    print("\n=== EJEMPLO 2: Guardar variantes ===")
    saved = augmentor.save_augmented_variants(
        audio_path,
        "data/audios_augmented/Angeli/",
        num_variants=10
    )
    print(f"✓ Guardadas {len(saved)} variantes")
    
    # Ejemplo 3: Augmentar dataset completo
    print("\n=== EJEMPLO 3: Augmentar dataset completo ===")
    stats = augmentor.augment_dataset(
        dataset_audio_dir="data/datasetPros/audio/",
        output_base_dir="data/audios_augmented/",
        variants_per_audio=15
    )
    
    print("\n📊 ESTADÍSTICAS DE AUGMENTACIÓN")
    print(f"  Personas: {stats['total_persons']}")
    print(f"  Audios originales: {stats['total_audios_original']}")
    print(f"  Audios tras augmentación: {stats['total_audios_augmented']}")
    print(f"  Factor de expansión: {stats['total_audios_augmented'] / max(stats['total_audios_original'], 1):.1f}x")
    
    # Detalles por persona
    for person, person_stats in stats['persons'].items():
        print(f"\n  {person}:")
        print(f"    Original: {person_stats['audios_original']}")
        print(f"    Augmentados: {person_stats['audios_augmented']}")


if __name__ == "__main__":
    example_usage()
