"""
Módulo para la detección y extracción de audio/voz de videos.

Implementa funcionalidades de extracción de audio de videos,
detección de voz y preprocesamiento para características MFCC.
"""

import cv2
import numpy as np
import librosa
import librosa.display
import os
from typing import Optional, Tuple, List, Dict


class AudioDetection:
    """
    Detector y extractor de audio/voz de videos.
    
    Extrae el stream de audio de un video, detecta presencia de voz
    y lo prepara para extracción de características MFCC.
    """
    
    def __init__(self, sr: int = 22050):
        """
        Inicializa el detector de audio.
        
        Args:
            sr (int): Frecuencia de muestreo en Hz. Default: 22050.
        """
        self.sr = sr
        self.audio_data = None
        self.video_path = None
        
        print(f"[AudioDetection] ✓ Detector inicializado - sr={sr}Hz")
    
    def extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extrae el stream de audio de un video.
        
        Args:
            video_path (str): Ruta del archivo de video.
        
        Returns:
            np.ndarray: Array de audio (mono, frecuencia sr).
                       None si no se pudo extraer audio.
        
        Raises:
            FileNotFoundError: Si el video no existe.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        try:
            # Abrir video con OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"[AudioDetection] ⚠️ No se pudo abrir: {video_path}")
                return None
            
            # Intentar extraer audio usando librosa (primera opción)
            try:
                audio, sr = librosa.load(video_path, sr=self.sr, mono=True)
                self.audio_data = audio
                self.video_path = video_path
                print(f"[AudioDetection] ✓ Audio extraído - {len(audio)} muestras, {sr}Hz")
                cap.release()
                return audio
            except Exception as e_lib:
                print(f"[AudioDetection] ⚠️ Error extrayendo audio con librosa: {e_lib}")
                # Intentar extraer audio con ffmpeg como fallback
                try:
                    import shutil
                    import tempfile
                    import subprocess

                    ffmpeg_bin = shutil.which('ffmpeg')
                    if ffmpeg_bin is None:
                        raise RuntimeError('ffmpeg no disponible en PATH')

                    tmp_fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
                    os.close(tmp_fd)

                    cmd = [ffmpeg_bin, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', str(self.sr), '-ac', '1', tmp_wav]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    audio, sr = librosa.load(tmp_wav, sr=self.sr, mono=True)
                    self.audio_data = audio
                    self.video_path = video_path

                    # Limpiar archivo temporal
                    try:
                        os.remove(tmp_wav)
                    except Exception:
                        pass

                    print(f"[AudioDetection] ✓ Audio extraído vía ffmpeg - {len(audio)} muestras, {sr}Hz")
                    cap.release()
                    return audio

                except Exception as e_ff:
                    print(f"[AudioDetection] ⚠️ Error extrayendo audio con ffmpeg fallback: {e_ff}")
                    cap.release()
                    return None
            
        except Exception as e:
            print(f"[AudioDetection] ❌ Error: {e}")
            return None
    
    def detect_voice_presence(self, audio_data: Optional[np.ndarray] = None,threshold: float = 0.02) -> Dict:
        """
        Detecta presencia de voz en el audio.
        
        Args:
            audio_data (np.ndarray): Datos de audio. Si None, usa self.audio_data.
            threshold (float): Umbral de energía para detectar voz (0-1).
        
        Returns:
            dict: {
                'has_voice': bool,
                'voice_energy': float,
                'silence_ratio': float,
                'num_voice_segments': int,
                'voice_frames': np.ndarray (frames con voz)
            }
        """
        if audio_data is None:
            audio_data = self.audio_data
        
        if audio_data is None or len(audio_data) == 0:
            return {
                'has_voice': False,
                'voice_energy': 0.0,
                'silence_ratio': 1.0,
                'num_voice_segments': 0,
                'voice_frames': None
            }
        
        # Calcular energía RMS por frame
        frame_length = 2048
        hop_length = 512
        
        S = librosa.feature.melspectrogram(y=audio_data, sr=self.sr, 
                                          n_fft=frame_length,
                                          hop_length=hop_length)
        
        # Convertir a dB
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Calcular energía media por frame
        energy = np.mean(S_db, axis=0)
        
        # Normalizar energía a rango 0-1
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-7)
        
        # Detectar frames con voz (energía > threshold)
        voice_frames = energy_norm > threshold
        num_voice_segments = np.sum(np.diff(voice_frames.astype(int)) > 0)
        
        # Calcular ratios
        voice_ratio = np.sum(voice_frames) / len(voice_frames)
        silence_ratio = 1.0 - voice_ratio
        
        has_voice = voice_ratio > 0.1  # Al menos 10% del audio es voz
        
        print(f"[AudioDetection] Análisis de voz:")
        print(f"  - Energía media: {np.mean(energy_norm):.3f}")
        print(f"  - Ratio voz: {voice_ratio:.1%}")
        print(f"  - Segmentos: {num_voice_segments}")
        
        return {
            'has_voice': has_voice,
            'voice_energy': float(np.mean(energy_norm)),
            'silence_ratio': float(silence_ratio),
            'num_voice_segments': int(num_voice_segments),
            'voice_frames': voice_frames
        }
    
    def segment_audio_by_voice(self, audio_data: Optional[np.ndarray] = None,
                              threshold: float = 0.02) -> List[np.ndarray]:
        """
        Segmenta el audio en partes con voz.
        
        Args:
            audio_data (np.ndarray): Datos de audio.
            threshold (float): Umbral de energía.
        
        Returns:
            list: Lista de segmentos de audio con voz.
        """
        if audio_data is None:
            audio_data = self.audio_data
        
        if audio_data is None or len(audio_data) == 0:
            return []
        
        # Detectar frames con voz
        detection = self.detect_voice_presence(audio_data, threshold)
        voice_frames = detection['voice_frames']
        
        if voice_frames is None:
            return [audio_data]
        
        # Convertir frames a muestras
        frame_length = 2048
        hop_length = 512
        
        # Encontrar límites de segmentos con voz
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_segment:
                # Inicio de segmento
                segment_start = i * hop_length
                in_segment = True
            elif not is_voice and in_segment:
                # Fin de segmento
                segment_end = i * hop_length
                segment = audio_data[segment_start:segment_end]
                if len(segment) > 0:
                    segments.append(segment)
                in_segment = False
        
        # Capturar último segmento si está en curso
        if in_segment:
            segment = audio_data[segment_start:]
            if len(segment) > 0:
                segments.append(segment)
        
        if not segments:
            segments = [audio_data]
        
        print(f"[AudioDetection] {len(segments)} segmentos de voz extraídos")
        return segments
    
    def extract_audio_features_info(self, audio_data: Optional[np.ndarray] = None) -> Dict:
        """
        Extrae información de características del audio.
        
        Args:
            audio_data (np.ndarray): Datos de audio.
        
        Returns:
            dict: Información sobre el audio (duración, duración con voz, etc.)
        """
        if audio_data is None:
            audio_data = self.audio_data
        
        if audio_data is None or len(audio_data) == 0:
            return {
                'duration_seconds': 0.0,
                'num_samples': 0,
                'sample_rate': self.sr
            }
        
        duration = len(audio_data) / self.sr
        
        # Detectar voz
        voice_detection = self.detect_voice_presence(audio_data)
        voice_duration = duration * (1 - voice_detection['silence_ratio'])
        
        return {
            'duration_seconds': float(duration),
            'voice_duration_seconds': float(voice_duration),
            'num_samples': len(audio_data),
            'sample_rate': self.sr,
            'has_voice': voice_detection['has_voice'],
            'voice_energy': voice_detection['voice_energy']
        }
    
    def process_video(self, video_path: str, save_audio: Optional[str] = None) -> Dict:
        """
        Procesa un video: extrae audio y lo prepara para MFCC.
        
        Args:
            video_path (str): Ruta del video.
            save_audio (str): Opcional, ruta donde guardar el audio extraído.
        
        Returns:
            dict: {
                'audio': np.ndarray,
                'voice_detection': dict,
                'audio_info': dict,
                'success': bool
            }
        """
        try:
            # Extraer audio
            audio = self.extract_audio_from_video(video_path)
            
            if audio is None or len(audio) == 0:
                return {
                    'audio': None,
                    'voice_detection': None,
                    'audio_info': None,
                    'success': False,
                    'error': 'No se pudo extraer audio'
                }
            
            # Detectar voz
            voice_detection = self.detect_voice_presence(audio)
            
            # Obtener información
            audio_info = self.extract_audio_features_info(audio)
            
            # Guardar audio si se especifica
            if save_audio:
                os.makedirs(os.path.dirname(save_audio), exist_ok=True)
                # Preferir soundfile si está disponible, sino fallback a scipy
                try:
                    import soundfile as sf
                    sf.write(save_audio, audio, self.sr, subtype='PCM_16')
                    print(f"[AudioDetection] ✓ Audio guardado (soundfile): {save_audio}")
                except Exception:
                    try:
                        from scipy.io import wavfile
                        wav_int16 = (audio * 32767).astype('int16')
                        wavfile.write(save_audio, self.sr, wav_int16)
                        print(f"[AudioDetection] ✓ Audio guardado (scipy): {save_audio}")
                    except Exception as e:
                        print(f"[AudioDetection] ❌ Error guardando audio: {e}")
                        raise
            
            return {
                'audio': audio,
                'voice_detection': voice_detection,
                'audio_info': audio_info,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            print(f"[AudioDetection] ❌ Error procesando video: {e}")
            return {
                'audio': None,
                'voice_detection': None,
                'audio_info': None,
                'success': False,
                'error': str(e)
            }
    
    def validate_audio_for_mfcc(self, audio_data: Optional[np.ndarray] = None,
                               min_voice_ratio: float = 0.05) -> Dict:
        """
        Valida que el audio sea adecuado para extracción MFCC.
        
        Args:
            audio_data (np.ndarray): Datos de audio.
            min_voice_ratio (float): Ratio mínimo de voz requerido.
        
        Returns:
            dict: {
                'is_valid': bool,
                'reason': str,
                'voice_ratio': float,
                'recommendation': str
            }
        """
        if audio_data is None:
            audio_data = self.audio_data
        
        if audio_data is None or len(audio_data) == 0:
            return {
                'is_valid': False,
                'reason': 'Audio vacío',
                'voice_ratio': 0.0,
                'recommendation': 'Capturar audio válido'
            }
        
        # Detectar voz
        voice_detection = self.detect_voice_presence(audio_data)
        voice_ratio = 1 - voice_detection['silence_ratio']
        
        # Validaciones
        issues = []
        
        if not voice_detection['has_voice']:
            issues.append('No hay voz detectada')
        
        if voice_ratio < min_voice_ratio:
            issues.append(f'Ratio de voz ({voice_ratio:.1%}) < mínimo ({min_voice_ratio:.1%})')
        
        if len(audio_data) < self.sr:  # Menos de 1 segundo
            issues.append('Audio muy corto (< 1 segundo)')
        
        is_valid = len(issues) == 0
        
        if is_valid:
            reason = "Audio válido para MFCC"
            recommendation = "Proceder con extracción MFCC"
        else:
            reason = " | ".join(issues)
            recommendation = "Considerar recapturar audio"
        
        return {
            'is_valid': is_valid,
            'reason': reason,
            'voice_ratio': float(voice_ratio),
            'recommendation': recommendation
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def example_usage():
    """Ejemplo de cómo usar AudioDetection."""
    
    # Crear detector
    audio_detector = AudioDetection(sr=22050)
    
    # Procesar video (ejemplo)
    video_path = "videos/person_001.mp4"
    
    result = audio_detector.process_video(video_path)
    
    if result['success']:
        print("\n✓ Procesamiento exitoso!")
        
        # Información del audio
        print(f"\nInformación del audio:")
        for key, value in result['audio_info'].items():
            print(f"  {key}: {value}")
        
        # Detección de voz
        print(f"\nDetección de voz:")
        for key, value in result['voice_detection'].items():
            if key != 'voice_frames':
                print(f"  {key}: {value}")
        
        # Validar para MFCC
        validation = audio_detector.validate_audio_for_mfcc(result['audio'])
        print(f"\nValidación para MFCC:")
        print(f"  Válido: {validation['is_valid']}")
        print(f"  Razón: {validation['reason']}")
        print(f"  Recomendación: {validation['recommendation']}")
        
        # Usar audio para MFCC
        if validation['is_valid']:
            from feature_extraction.mfcc import MFCCExtractor
            
            mfcc_extractor = MFCCExtractor()
            mfcc_features = mfcc_extractor.extract(result['audio'])
            
            print(f"\n✓ MFCC extraído - Shape: {mfcc_features.shape}")
    
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    example_usage()
