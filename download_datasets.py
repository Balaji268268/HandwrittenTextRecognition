"""
Dataset Downloader and Manager for PySarah Handwritten Text Recognition.
Supports all integrated datasets with automatic Google Drive / Zenodo downloads.
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path

# All supported dataset sources with their Google Drive IDs and configurations
DATASETS = {
    'washington': {
        'desc': 'Washington handwriting database (English)',
        'id': '1MuKc3D3SoWVUJPYqhmnOT9Xd-CWnmpk4',
        'url': 'https://drive.google.com/uc?id=1MuKc3D3SoWVUJPYqhmnOT9Xd-CWnmpk4',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'washington.zip'
    },
    'mnist': {
        'desc': 'MNIST isolated handwritten digits (0-9)',
        'id': '1QwW7RgxQjJ83nJQ5dMw4mWawyCcpwRp8',
        'url': 'https://drive.google.com/uc?id=1QwW7RgxQjJ83nJQ5dMw4mWawyCcpwRp8',
        'text_level': 'char',
        'image_shape': (28, 28, 1),
        'filename': 'mnist.zip'
    },
    'saintgall': {
        'desc': 'Saint Gall medieval manuscript (Latin)',
        'id': '1X66lsJFEK-RixO4dQ0DP9ZH6yVc64P2m',
        'url': 'https://drive.google.com/uc?id=1X66lsJFEK-RixO4dQ0DP9ZH6yVc64P2m',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'saintgall.zip'
    },
    'parzival': {
        'desc': 'Parzival medieval manuscript (Middle High German)',
        'id': '1szhdRxYRCkkIehQaLgTk3YD9At563BSU',
        'url': 'https://drive.google.com/uc?id=1szhdRxYRCkkIehQaLgTk3YD9At563BSU',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'parzival.zip'
    },
    'cvl-digits': {
        'desc': 'CVL Digits string database',
        'id': '1jPZtUtiARgrbFaCfPyPg_UZrgwYHJ7Td',
        'url': 'https://drive.google.com/uc?id=1jPZtUtiARgrbFaCfPyPg_UZrgwYHJ7Td',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'cvl-digits.zip'
    },
    'emnist': {
        'desc': 'Extended MNIST handwritten characters',
        'id': '107Lzd-5fAkt3XLdOgTWH5yQeqd3QHbKf',
        'url': 'https://drive.google.com/uc?id=107Lzd-5fAkt3XLdOgTWH5yQeqd3QHbKf',
        'text_level': 'char',
        'image_shape': (28, 28, 1),
        'filename': 'emnist.zip'
    },
    'bentham': {
        'desc': 'Bentham manuscripts collection (English)',
        'id': '1do3tS7vd-QaUxkeBwPE4Phia99-J_AUq',
        'url': 'https://drive.google.com/uc?id=1do3tS7vd-QaUxkeBwPE4Phia99-J_AUq',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'bentham.zip'
    },
    'cvl-database': {
        'desc': 'CVL Database of handwritten words and lines',
        'id': '1H0M2lHdxUCLs7eKjk2q7fNBYU0-PkLDo',
        'url': 'https://drive.google.com/uc?id=1H0M2lHdxUCLs7eKjk2q7fNBYU0-PkLDo',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'cvl-database.zip'
    },
    'rimes': {
        'desc': 'RIMES French handwriting database',
        'id': '1iax6qNqKtg0PHZl68HnJy2WRFQ2IcRLM',
        'url': 'https://drive.google.com/uc?id=1iax6qNqKtg0PHZl68HnJy2WRFQ2IcRLM',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'rimes.zip'
    },
    'iam': {
        'desc': 'IAM Handwriting database (English)',
        'id': '1z6gOT4U_eTsCguSCWAz3IaQwTD8TXwDw',
        'url': 'https://drive.google.com/uc?id=1z6gOT4U_eTsCguSCWAz3IaQwTD8TXwDw',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'iam.zip'
    },
    'orand-car': {
        'desc': 'ORAND-CAR handwritten digit strings (used for orand-car-a and orand-car-b)',
        'id': '1jkT2ow85eob9hK4xygdOjlIT6zNMUlAG',
        'url': 'https://drive.google.com/uc?id=1jkT2ow85eob9hK4xygdOjlIT6zNMUlAG',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'orand-car.zip'
    },
    'bressay': {
        'desc': 'BRESSAY historical document database',
        'id': None,
        'url': 'https://zenodo.org/records/11637681/files/bressay.zip?download=1',
        'text_level': 'line',
        'image_shape': (64, 1024, 1),
        'filename': 'bressay.zip'
    }
}


def get_datasets_dir():
    """Returns the absolute path to the datasets directory."""
    datasets_dir = Path(__file__).resolve().parent
    if datasets_dir.name != 'datasets':
        datasets_dir = Path.cwd() / 'datasets'
    return datasets_dir


def check_status(name, datasets_dir=None):
    """
    Checks if a dataset is extracted, downloaded as zip, or missing.
    Returns: 'EXTRACTED', 'ZIP_READY', or 'MISSING'
    """
    if datasets_dir is None:
        datasets_dir = get_datasets_dir()

    info = DATASETS.get(name)
    if not info:
        return 'UNKNOWN'

    dir_path = datasets_dir / name
    zip_path = datasets_dir / info['filename']

    if dir_path.is_dir() and any(dir_path.iterdir()):
        return 'EXTRACTED'
    elif zip_path.is_file() and zip_path.stat().st_size > 1000:
        return 'ZIP_READY'
    return 'MISSING'


def list_datasets(datasets_dir=None):
    """Prints a formatted summary of all dataset sources and their local status."""
    if datasets_dir is None:
        datasets_dir = get_datasets_dir()

    print("=" * 78)
    print(f"{'Source':<14} {'Status':<13} {'Default Shape':<18} {'Description'}")
    print("-" * 78)

    for name, info in DATASETS.items():
        status = check_status(name, datasets_dir)
        status_str = f"[{status}]"
        shape_str = f"{info['image_shape']} ({info['text_level']})"
        desc = info['desc']
        if len(desc) > 30:
            desc = desc[:27] + "..."
        print(f"{name:<14} {status_str:<13} {shape_str:<18} {desc}")

    print("=" * 78)


def download_dataset(name, datasets_dir=None, force=False, auto_extract=True):
    """Downloads a dataset zip using gdown (or requests for Zenodo) and auto-extracts."""
    if datasets_dir is None:
        datasets_dir = get_datasets_dir()

    datasets_dir.mkdir(parents=True, exist_ok=True)

    if name not in DATASETS:
        print(f"Error: Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
        return False

    info = DATASETS[name]
    dest = datasets_dir / info['filename']

    if dest.exists() and not force:
        status = check_status(name, datasets_dir)
        if status in ('EXTRACTED', 'ZIP_READY'):
            print(f"Dataset '{name}' is already present at: {dest} (status: {status})")
            return True

    print(f"Downloading '{name}' to {dest}...")
    try:
        import gdown
        if info['id']:
            url = f"https://drive.google.com/uc?id={info['id']}"
            gdown.download(url, str(dest), quiet=False)
        else:
            import requests
            r = requests.get(info['url'], stream=True)
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded '{name}' successfully.")
        if auto_extract:
            extract_dataset(name, datasets_dir)
        return True
    except Exception as e:
        print(f"Error downloading '{name}': {e}")
        return False


def extract_dataset(name, datasets_dir=None):
    """Extracts a downloaded dataset zip file into the datasets directory."""
    if datasets_dir is None:
        datasets_dir = get_datasets_dir()

    info = DATASETS.get(name)
    if not info:
        print(f"Error: Unknown dataset '{name}'")
        return False

    zip_path = datasets_dir / info['filename']
    if not zip_path.is_file():
        print(f"Error: Zip file {zip_path} not found. Please download it first.")
        return False

    print(f"Extracting {zip_path.name} into {datasets_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(datasets_dir)
    print(f"Extracted '{name}' successfully.")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PySarah Dataset Downloader and Manager")
    parser.add_argument('--list', action='store_true', help="List all datasets and their local status")
    parser.add_argument('--download', type=str, help="Dataset name to download (e.g. washington, mnist, saintgall, parzival)")
    parser.add_argument('--download-all', action='store_true', help="Download all missing datasets")
    parser.add_argument('--extract', type=str, help="Extract an already-downloaded dataset zip")

    args = parser.parse_args()

    if args.list or len(sys.argv) == 1:
        list_datasets()
    elif args.download:
        download_dataset(args.download)
    elif args.download_all:
        for ds_name in DATASETS:
            status = check_status(ds_name)
            if status == 'MISSING':
                download_dataset(ds_name)
    elif args.extract:
        extract_dataset(args.extract)
