import os
import sys
import importlib


class Source():
    """
    Represents an aggregate of all English handwriting data sources:
    - washington (18th-century cursive English)
    - iam (Modern English handwriting benchmark)
    - bentham (18th-19th century English manuscripts)
    - cvl-database (Modern English handwriting by 310 writers)
    """

    def __init__(self, artifact_path):
        self.artifact_path = artifact_path

    def fetch_data(self, text_level):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        if curr_dir not in sys.path:
            sys.path.append(curr_dir)

        include_list = [
            'washington',
            'iam',
            'bentham',
            'cvl-database',
        ]

        data = {'training': [], 'validation': [], 'test': []}
        global_writer_mapping = {}
        global_writer_id = 1

        for source_name in include_list:
            source_dir = os.path.join(self.artifact_path, source_name)
            # Only load if the dataset exists locally
            if not os.path.isdir(source_dir):
                continue

            try:
                module = importlib.import_module(source_name)
                source = module.Source(self.artifact_path)
                source_data = source.fetch_data(text_level)

                local_writer_mapping = {}
                local_writer_id = 1

                for dataset_type in ['training', 'validation', 'test']:
                    for item in source_data[dataset_type]:
                        original_writer = item['writer']

                        if original_writer not in local_writer_mapping:
                            local_writer_mapping[original_writer] = local_writer_id
                            global_writer_mapping[local_writer_id] = global_writer_id

                            local_writer_id += 1
                            global_writer_id += 1

                        item['writer'] = str(global_writer_mapping[local_writer_mapping[original_writer]])

                    data[dataset_type].extend(source_data[dataset_type])
            except Exception as e:
                print(f"Warning: Failed to load source '{source_name}': {e}")

        return data
