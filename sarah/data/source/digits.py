import os
import sys
import glob
import random
from pathlib import Path


class Source():
    """
    Unified Handwritten Digits & Numbers database source.
    Combines:
      - CVL Digits (7,960 handwritten digit strings)
      - ORAND-CAR A (5,793 real check amount numbers)
      - ORAND-CAR B (5,926 real check amount numbers)
    Total: ~19,679 real handwritten number and digit string samples.
    """

    def __init__(self, artifact_path):
        """
        Initializes the Digits source.

        Parameters
        ----------
        artifact_path : str
            The path to the datasets directory.
        """
        self.artifact_path = os.path.abspath(artifact_path)
        self.base_path = os.path.join(self.artifact_path, 'cvl-digits')
        self._ensure_datasets()

    def _ensure_datasets(self):
        """
        Auto-download cvl-digits and orand-car if missing.
        """
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from download_datasets import download_dataset, check_status

            datasets_dir = Path(self.artifact_path)
            for ds in ['cvl-digits', 'orand-car']:
                status = check_status(ds, datasets_dir)
                if status == 'MISSING':
                    print(f"📥 Digits source: '{ds}' not found. Auto-downloading to {self.artifact_path}...")
                    download_dataset(ds, datasets_dir=datasets_dir, auto_extract=True)
                elif status == 'ZIP_READY':
                    from download_datasets import extract_dataset
                    print(f"📦 Extracting {ds}.zip...")
                    extract_dataset(ds, datasets_dir=datasets_dir)
        except Exception as e:
            print(f"Note: Digits source dataset verification: {e}")

    def fetch_data(self, text_level):
        """
        Retrieves the merged digit data partitioned into training, validation, and test.

        Parameters
        ----------
        text_level : str
            The granularity of the data ('line' or 'word').

        Returns
        -------
        dict
            Dictionary with 'training', 'validation', and 'test' lists.
        """
        data = {'training': [], 'validation': [], 'test': []}

        cvl_base = os.path.join(self.artifact_path, 'cvl-digits')
        car_base = os.path.join(self.artifact_path, 'orand-car')

        all_items = []

        # 1. Load CVL Digits
        if os.path.isdir(cvl_base):
            cvl_patterns = [
                os.path.join(cvl_base, 'cvl-strings', 'train', '**', '*.png'),
                os.path.join(cvl_base, 'cvl-strings-eval', '**', '*.png')
            ]
            for pattern in cvl_patterns:
                for img_path in glob.glob(pattern, recursive=True):
                    base_name = os.path.basename(img_path)
                    parts = base_name.split('-')
                    if len(parts) >= 2:
                        text = parts[0]
                        writer = f"cvl_{parts[1]}"
                        all_items.append({
                            'image': img_path,
                            'bbox': [],
                            'text': text,
                            'writer': writer
                        })

        # 2. Load ORAND-CAR A
        car_a_train = os.path.join(car_base, 'CAR-A', 'a_train_images')
        car_a_gt = os.path.join(car_base, 'CAR-A', 'a_train_gt.txt')
        if os.path.isfile(car_a_gt) and os.path.isdir(car_a_train):
            with open(car_a_gt, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        p = os.path.join(car_a_train, parts[0])
                        if os.path.isfile(p):
                            all_items.append({
                                'image': p,
                                'bbox': [],
                                'text': parts[1],
                                'writer': 'car_a'
                            })

        car_a_test = os.path.join(car_base, 'CAR-A', 'a_test_images')
        car_a_test_gt = os.path.join(car_base, 'CAR-A', 'a_test_gt.txt')
        if os.path.isfile(car_a_test_gt) and os.path.isdir(car_a_test):
            with open(car_a_test_gt, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        p = os.path.join(car_a_test, parts[0])
                        if os.path.isfile(p):
                            all_items.append({
                                'image': p,
                                'bbox': [],
                                'text': parts[1],
                                'writer': 'car_a'
                            })

        # 3. Load ORAND-CAR B
        car_b_train = os.path.join(car_base, 'CAR-B', 'b_train_images')
        car_b_gt = os.path.join(car_base, 'CAR-B', 'b_train_gt.txt')
        if os.path.isfile(car_b_gt) and os.path.isdir(car_b_train):
            with open(car_b_gt, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        p = os.path.join(car_b_train, parts[0])
                        if os.path.isfile(p):
                            all_items.append({
                                'image': p,
                                'bbox': [],
                                'text': parts[1],
                                'writer': 'car_b'
                            })

        car_b_test = os.path.join(car_base, 'CAR-B', 'b_test_images')
        car_b_test_gt = os.path.join(car_base, 'CAR-B', 'b_test_gt.txt')
        if os.path.isfile(car_b_test_gt) and os.path.isdir(car_b_test):
            with open(car_b_test_gt, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        p = os.path.join(car_b_test, parts[0])
                        if os.path.isfile(p):
                            all_items.append({
                                'image': p,
                                'bbox': [],
                                'text': parts[1],
                                'writer': 'car_b'
                            })

        # Deterministic shuffle and partition (80% train, 10% val, 10% test)
        rng = random.Random(42)
        rng.shuffle(all_items)

        total = len(all_items)
        if total > 0:
            n_val = max(1, int(total * 0.10))
            n_test = max(1, int(total * 0.10))
            n_train = total - n_val - n_test

            data['training'] = all_items[:n_train]
            data['validation'] = all_items[n_train:n_train + n_val]
            data['test'] = all_items[n_train + n_val:]

        return data

