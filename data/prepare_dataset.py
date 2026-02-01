"""
Utilities to create dataset structure and optionally auto-distribute videos
from a `data/raw` folder into `data/dataset/{person}/{front,back}`.

Usage examples:
  python -m data.prepare_dataset --create persons.txt
  python -m data.prepare_dataset --auto-distribute

The script is tolerant to filenames containing: 'frente', 'front', 'espalda', 'back'
and extracts the person name by splitting by those keywords or using the
filename prefix.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}


def create_structure(dataset_path: str = 'data/dataset', persons: Optional[List[str]] = None):
    """Create the empty dataset structure.

    Creates folders: data/dataset/{person}/{front,back}
    If persons is None, creates a placeholder 'unknown' folder.
    """
    base = Path(dataset_path)
    base.mkdir(parents=True, exist_ok=True)

    if not persons:
        persons = ['unknown']

    for p in persons:
        person_dir = base / p
        (person_dir / 'front').mkdir(parents=True, exist_ok=True)
        (person_dir / 'back').mkdir(parents=True, exist_ok=True)

    print(f"✓ Estructura creada en: {base}")


def _detect_view_and_person(filename: str):
    name = filename.lower()
    view = None
    if 'frente' in name or 'front' in name:
        view = 'front'
    elif 'espalda' in name or 'back' in name:
        view = 'back'

    # Try to extract person name before the keyword
    if view:
        keyword = 'frente' if 'frente' in name else ('front' if 'front' in name else ('espalda' if 'espalda' in name else 'back'))
        parts = filename.split(keyword, 1)
        person = parts[0].strip().replace('_', '').replace('-', '').replace(' ', '')
        if person == '':
            person = 'unknown'
    else:
        # fallback: prefix up to first non-alpha numeric
        person = ''.join(ch for ch in Path(filename).stem if ch.isalnum())
        view = 'unknown'

    return person, view


def auto_distribute(raw_folder: str = 'data/raw', dataset_folder: str = 'data/dataset', move_files: bool = False):
    """Scan raw folder and distribute video files into dataset structure.

    If move_files is True, files are moved; otherwise they are copied.
    """
    raw = Path(raw_folder)
    dst = Path(dataset_folder)
    if not raw.exists():
        raise FileNotFoundError(f"Raw folder not found: {raw}")

    for f in raw.iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            person, view = _detect_view_and_person(f.name)
            if view not in ['front', 'back']:
                print(f"⚠️ No se pudo determinar view para: {f.name}, saltando")
                continue

            target_dir = dst / person / view
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f.name

            if move_files:
                shutil.move(str(f), str(target_path))
                action = 'movido'
            else:
                shutil.copy2(str(f), str(target_path))
                action = 'copiado'

            print(f"{action}: {f.name} -> {target_dir}")

    print("✓ Distribución completada.")


if __name__ == '__main__':
    # Simple CLI for local use
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataset folders and distribute raw videos')
    parser.add_argument('--create', nargs='*', help='Lista de personas a crear (separadas por espacio)')
    parser.add_argument('--auto-distribute', action='store_true', help='Distribuye archivos desde data/raw')
    parser.add_argument('--move', action='store_true', help='Mover archivos en lugar de copiar')
    args = parser.parse_args()

    if args.create:
        create_structure(persons=args.create)
    if args.auto_distribute:
        auto_distribute(move_files=args.move)
    if not args.create and not args.auto_distribute:
        parser.print_help()