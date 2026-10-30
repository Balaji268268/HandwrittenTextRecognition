import os
import sys
import importlib


class Source():
    """
    Combined IAM (Sentences, Words, Lines) + Digits (CVL-digits + ORAND-CAR A & B).
    Total: ~91,000+ samples.
    Provides complete English handwriting recognition covering letters, punctuation,
    and high-density handwritten numbers/digits with zero catastrophic forgetting.
    """

    def __init__(self, artifact_path):
        """
        Initializes the IAM-Digits source.

        Parameters
        ----------
        artifact_path : str
            The path to the datasets directory.
        """
        self.artifact_path = os.path.abspath(artifact_path)
        self.base_path = os.path.join(self.artifact_path, 'iam')

    def fetch_data(self, text_level):
        """
        Retrieves the combined dataset partitioned into training, validation, and test.

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

        # 1. Load IAM
        try:
            iam_mod = importlib.import_module('sarah.data.source.iam')
            iam_src = iam_mod.Source(self.artifact_path)
            iam_data = iam_src.fetch_data(text_level)

            for part in ['training', 'validation', 'test']:
                for item in iam_data.get(part, []):
                    item_copy = dict(item)
                    item_copy['writer'] = f"iam_{item_copy.get('writer', '1')}"
                    data[part].append(item_copy)
        except Exception as e:
            print(f"Warning: Failed to load IAM component in iam-digits: {e}")

        # 2. Load Digits (CVL-digits + ORAND-CAR)
        try:
            digits_mod = importlib.import_module('sarah.data.source.digits')
            digits_src = digits_mod.Source(self.artifact_path)
            digits_data = digits_src.fetch_data(text_level)

            for part in ['training', 'validation', 'test']:
                for item in digits_data.get(part, []):
                    item_copy = dict(item)
                    item_copy['writer'] = f"digit_{item_copy.get('writer', '1')}"
                    data[part].append(item_copy)
        except Exception as e:
            print(f"Warning: Failed to load Digits component in iam-digits: {e}")

        print(f"📊 [IAM + Digits] Loaded: Training={len(data['training']):,}, "
              f"Validation={len(data['validation']):,}, Test={len(data['test']):,}")

        return data
