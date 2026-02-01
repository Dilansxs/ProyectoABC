"""
Extrae y guarda audios de los videos BACK de cada persona del dataset.

Guarda archivos WAV en `data/datasetPros/audio/{person}/{video_basename}.wav`.
"""
import os
from pathlib import Path
from typing import Optional
from detection.audio_detection import AudioDetection


class AudioExtractor:
    def __init__(self, dataset_path: str = 'data/dataset', output_path: str = 'data/datasetPros'):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.detector = AudioDetection(video_type='back')

    def process_back_videos(self, save_audio: Optional[bool] = True) -> dict:
        stats = {'videos_processed': 0, 'audios_saved': 0, 'errors': []}

        for person_dir in sorted(self.dataset_path.iterdir()):
            if not person_dir.is_dir():
                continue
            back_dir = person_dir / 'back'
            if not back_dir.exists():
                continue

            for video in back_dir.iterdir():
                if not video.is_file():
                    continue
                try:
                    out_audio_dir = self.output_path / 'audio' / person_dir.name
                    out_audio_dir.mkdir(parents=True, exist_ok=True)
                    out_audio_file = out_audio_dir / (video.stem + '.wav')

                    result = self.detector.process_video(str(video), save_audio=str(out_audio_file) if save_audio else None)
                    stats['videos_processed'] += 1
                    if result.get('success') and save_audio:
                        stats['audios_saved'] += 1
                    if not result.get('success'):
                        stats['errors'].append({'video': str(video), 'error': result.get('error')})

                    print(f"Procesado: {video.name} - éxito: {result.get('success')}")
                except Exception as e:
                    stats['errors'].append({'video': str(video), 'error': str(e)})

        return stats


if __name__ == '__main__':
    extractor = AudioExtractor()
    stats = extractor.process_back_videos()
    print("\nResumen:")
    print(stats)