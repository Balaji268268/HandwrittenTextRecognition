import os
import glob


class Source():
    """
    Represents the IAM database source.

    Requires implementation of `fetch_data`, returning a dictionary with
        'training', 'validation', and 'test' keys, each mapping to a list of data entries.

    Each data entry is a dictionary with keys 'id', 'image', 'bbox', 'text', and 'writer':
        'id' : str
            unique item ID.
        'form_id' : str
            parent form ID.
        'image' : str
            path to the image file.
        'bbox' : list
            bounding box coordinates [x, y, w, h] (empty if pre-cropped).
        'text' : str
            text transcription.
        'writer' : str
            writer's unique ID.
    """

    def __init__(self, artifact_path):
        """
        Initializes a new instance of the Source class.

        Parameters
        ----------
        artifact_path : str
            The path to the data.
        """

        self.artifact_path = artifact_path
        self.base_path = os.path.join(self.artifact_path, 'iam')

        self.partition_path = os.path.join(self.base_path, 'largeWriterIndependentTextLineRecognitionTask')
        self.training_file_path = os.path.join(self.partition_path, 'trainset.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'validationset1.txt')
        self.test_file_path = os.path.join(self.partition_path, 'testset.txt')

        self.transcription_path = os.path.join(self.base_path, 'ascii')
        self.words_file_path = os.path.join(self.transcription_path, 'words.txt')
        self.lines_file_path = os.path.join(self.transcription_path, 'lines.txt')
        self.sentences_file_path = os.path.join(self.transcription_path, 'sentences.txt')
        self.forms_file_path = os.path.join(self.transcription_path, 'forms.txt')

        self.writers = self._get_writers(self.forms_file_path)
        self.image_cache = {}
        if os.path.isdir(self.base_path):
            self._index_images()

    def _index_images(self):
        """
        Index image files by stem (filename without extension) across all directories
        (lines, sentences, words, formsA-D, formsE-H, formsI-Z, forms, etc.)
        for O(1) instant resolution.
        """
        if not os.path.isdir(self.base_path):
            return

        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        skip_dirs = {'ascii', 'xml', 'largewriterindependenttextlinerecognitiontask'}

        for root, dirs, files in os.walk(self.base_path):
            if os.path.basename(root).lower() in skip_dirs:
                continue
            for f in files:
                name_lower = f.lower()
                if any(name_lower.endswith(ext) for ext in valid_exts):
                    stem = os.path.splitext(f)[0]
                    self.image_cache[stem] = os.path.join(root, f)

    def _resolve_image(self, item_id, form_id, default_bbox):
        """
        Returns (image_path, bbox).
        If cropped image exists (item_id in image_cache), returns (path, []).
        Else if full form image exists (form_id in image_cache), returns (form_path, default_bbox).
        Else returns (None, None).
        """
        direct_path = self.image_cache.get(item_id)
        if direct_path and os.path.isfile(direct_path):
            return direct_path, []

        form_path = self.image_cache.get(form_id)
        if form_path and os.path.isfile(form_path):
            return form_path, default_bbox

        return None, None

    def fetch_data(self, text_level):
        """
        Retrieves the data for training, validation, and test partitions.

        Parameters
        ----------
        text_level : str
            The granularity of the data to be retrieved.

        Returns
        -------
        dict
            Partition dictionary with list of items.
        """

        data = {'training': [], 'validation': [], 'test': []}

        if not os.path.isdir(self.base_path):
            return data

        if not self.image_cache:
            self._index_images()

        if not self.writers and os.path.isfile(self.forms_file_path):
            self.writers = self._get_writers(self.forms_file_path)

        training_partition_data = self._read_file(self.training_file_path)
        validation_partition_data = self._read_file(self.validation_file_path)
        test_partition_data = self._read_file(self.test_file_path)

        if text_level == 'word':
            words_data = self._load_words_data(self.words_file_path)

            data['training'] = self._get_data(words_data, training_partition_data)
            data['validation'] = self._get_data(words_data, validation_partition_data)
            data['test'] = self._get_data(words_data, test_partition_data)

        elif text_level == 'line':
            lines_data = self._load_lines_data(self.lines_file_path)

            data['training'] = self._get_data(lines_data, training_partition_data)
            data['validation'] = self._get_data(lines_data, validation_partition_data)
            data['test'] = self._get_data(lines_data, test_partition_data)

        elif text_level == 'sentence':
            sentences_data = self._load_sentences_data(self.sentences_file_path)

            data['training'] = self._get_data(sentences_data, training_partition_data)
            data['validation'] = self._get_data(sentences_data, validation_partition_data)
            data['test'] = self._get_data(sentences_data, test_partition_data)

        elif text_level == 'paragraph':
            lines_data = self._load_lines_data(self.lines_file_path)
            paragraphs_data = self._load_paragraphs_data(lines_data)

            data['training'] = self._get_data(paragraphs_data, training_partition_data, group=True)
            data['validation'] = self._get_data(paragraphs_data, validation_partition_data, group=True)
            data['test'] = self._get_data(paragraphs_data, test_partition_data, group=True)

        elif text_level in ['combined', 'all']:
            lines_data = self._load_lines_data(self.lines_file_path)
            sentences_data = self._load_sentences_data(self.sentences_file_path)
            words_data = self._load_words_data(self.words_file_path)

            for part, part_data in [('training', training_partition_data),
                                    ('validation', validation_partition_data),
                                    ('test', test_partition_data)]:
                p_lines = self._get_data(lines_data, part_data)
                p_sents = self._get_data(sentences_data, part_data)
                p_words = self._get_data(words_data, part_data)

                if part == 'test':
                    data[part] = p_lines
                else:
                    data[part] = p_lines + p_sents + p_words

        return data

    def _read_file(self, file_path):
        """
        Loads the data content from file.

        Parameters
        ----------
        file_path : str
            The path to the file content.

        Returns
        -------
        list
            A list of rows.
        """

        if not file_path or not os.path.isfile(file_path):
            return []

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = [x.strip() for x in f.readlines() if not x.startswith('#')]

        return data

    def _get_data(self, data, partition_data, group=False):
        """
        Filter the given data based on partition data.
        Matches exact item IDs or form IDs (for sentences/words/paragraphs).

        Parameters
        ----------
        data : list of dict
            The data to filter, each with 'id' and 'form_id'.
        partition_data : list
            The IDs to match against.
        group : bool, optional
            Indicates whether to group by form ID.

        Returns
        -------
        list of dict
            The filtered data matching the partition IDs.
        """

        if not data or not partition_data:
            return []

        exact_set = set(partition_data)
        form_set = {'-'.join(x.split('-')[:2]) for x in partition_data}

        filtered = []
        for x in data:
            item_id = x.get('id', '')
            form_id = x.get('form_id') or '-'.join(item_id.split('-')[:2])

            if group:
                if form_id in form_set:
                    filtered.append(x)
            else:
                if item_id in exact_set or form_id in form_set:
                    filtered.append(x)

        return filtered

    def _get_writers(self, forms_file_path):
        """
        Parses a file to assign unique IDs to each writer.

        Parameters
        ----------
        forms_file_path : str
            The forms filepath with the writers IDs.

        Returns
        -------
        dict
            Maps form IDs to unique writer IDs.
        """

        writers = {}
        data = self._read_file(forms_file_path)

        for row in data:
            row = row.split()

            if len(row) > 1:
                form_id, writer_id = row[0], row[1]
                writers[form_id] = writer_id

        return writers

    def _load_words_data(self, file_path):
        """
        Loads the words data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the words data.

        Returns
        -------
        list
            A list of words data.
        """

        words_data = []
        data = self._read_file(file_path)

        for row in data:
            row = row.split()
            if len(row) < 9:
                continue

            # Skip segmentation errors ('err')
            if row[1] != 'ok':
                continue

            ids = row[0].split('-')
            form_id = '-'.join(ids[:2])
            item_id = row[0]

            writer_id = self.writers.get(form_id, '1')
            try:
                bbox = [int(x) for x in row[3:7]]
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
            except Exception:
                bbox = []

            text = ' '.join(row[-1].replace('|', ' ').replace('...', '. . .').split())
            if not text:
                continue

            image_path, final_bbox = self._resolve_image(item_id, form_id, bbox)
            if not image_path:
                continue

            words_data.append({
                'id': item_id,
                'form_id': form_id,
                'image': image_path,
                'bbox': final_bbox,
                'text': text,
                'writer': writer_id,
            })

        return words_data

    def _load_lines_data(self, file_path):
        """
        Loads the lines data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the lines data.

        Returns
        -------
        list
            A list of lines data.
        """

        exclude_data = ['p02-109-01']

        lines_data = []
        data = self._read_file(file_path)

        for row in data:
            row = row.split()
            if len(row) < 8:
                continue

            ids = row[0].split('-')
            form_id = '-'.join(ids[:2])
            item_id = row[0]

            if item_id in exclude_data:
                continue

            writer_id = self.writers.get(form_id, '1')
            try:
                bbox = [int(x) for x in row[4:8]]
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
            except Exception:
                bbox = []

            transcription = ' '.join(row[8:]) if len(row) > 8 else row[-1]
            text = ' '.join(transcription.replace('|', ' ').replace('...', '. . .').split())
            if not text:
                continue

            image_path, final_bbox = self._resolve_image(item_id, form_id, bbox)
            if not image_path:
                continue

            lines_data.append({
                'id': item_id,
                'form_id': form_id,
                'image': image_path,
                'bbox': final_bbox,
                'text': text,
                'writer': writer_id,
            })

        return lines_data

    def _load_sentences_data(self, file_path):
        """
        Loads the sentences data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the sentences data.

        Returns
        -------
        list
            A list of sentences data.
        """

        sentences_data = []
        data = self._read_file(file_path)

        for row in data:
            row = row.split()
            if len(row) < 9:
                continue

            # Skip segmentation errors
            if 'err' in row[1:3]:
                continue

            ids = row[0].split('-')
            form_id = '-'.join(ids[:2])
            item_id = row[0]

            writer_id = self.writers.get(form_id, '1')
            try:
                bbox = [int(x) for x in row[5:9]]
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
            except Exception:
                bbox = []

            transcription = ' '.join(row[9:]) if len(row) > 9 else row[-1]
            text = ' '.join(transcription.replace('|', ' ').replace('...', '. . .').split())
            if not text:
                continue

            image_path, final_bbox = self._resolve_image(item_id, form_id, bbox)
            if not image_path:
                continue

            sentences_data.append({
                'id': item_id,
                'form_id': form_id,
                'image': image_path,
                'bbox': final_bbox,
                'text': text,
                'writer': writer_id,
            })

        return sentences_data

    def _load_paragraphs_data(self, lines_data):
        """
        Loads the paragraphs data.

        Parameters
        ----------
        lines_data : list
            Line data as base content for paragraph data.

        Returns
        -------
        list
            A list of paragraphs data.
        """

        paragraphs_data = {}

        for line in lines_data:
            form_id = line.get('form_id') or '-'.join(line['id'].split('-')[:2])
            form_path = self.image_cache.get(form_id)
            if not form_path:
                continue

            bbox = [
                line['bbox'][0] if line['bbox'] else 0,
                line['bbox'][1] if line['bbox'] else 0,
                (line['bbox'][2] + line['bbox'][0]) if line['bbox'] else 0,
                (line['bbox'][3] + line['bbox'][1]) if line['bbox'] else 0,
            ]

            content = {
                'id': form_id,
                'form_id': form_id,
                'image': form_path,
                'bbox': bbox,
                'text': '',
                'writer': line['writer'],
            }

            paragraph = paragraphs_data.setdefault(form_path, content)

            paragraph['bbox'] = [
                min(paragraph['bbox'][0], bbox[0]),
                min(paragraph['bbox'][1], bbox[1]),
                max(paragraph['bbox'][2], bbox[2]),
                max(paragraph['bbox'][3], bbox[3])
            ]

            paragraph['text'] += (f"\n{line['text']}" if paragraph['text'] else line['text'])

        paragraphs_data = list(paragraphs_data.values())

        for paragraph in paragraphs_data:
            paragraph['bbox'] = [
                paragraph['bbox'][0],
                paragraph['bbox'][1],
                abs(paragraph['bbox'][2] - paragraph['bbox'][0]),
                abs(paragraph['bbox'][3] - paragraph['bbox'][1])
            ]

        return paragraphs_data
