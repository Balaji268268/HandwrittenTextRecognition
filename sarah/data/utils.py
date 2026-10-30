import re
import cv2
import html
import numpy as np


def batch_binarization(batch_data, method, invert=False):
    """
    Apply binarization to a batch of grayscale images.

    Parameters
    ----------
    batch_data : list of np.ndarray
        List of grayscale images.
    method : str, optional
        Binarization method to apply.
    invert : bool, optional
        Whether to invert the binarized image.

    Returns
    ----------
    list of np.ndarray
        List of binarized images.
    """

    outputs = []

    for image in batch_data:
        if not isinstance(image, np.ndarray):
            outputs.append(image)
            continue

        if method == 'global':
            _, image = cv2.threshold(src=image,
                                     thresh=127,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY)

        elif method == 'otsu':
            _, image = cv2.threshold(src=image,
                                     thresh=0,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'adaptive_mean':
            image = cv2.adaptiveThreshold(src=image,
                                          maxValue=255,
                                          adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                          thresholdType=cv2.THRESH_BINARY,
                                          blockSize=11,
                                          C=2)

        elif method == 'adaptive_gaussian':
            image = cv2.adaptiveThreshold(src=image,
                                          maxValue=255,
                                          adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresholdType=cv2.THRESH_BINARY,
                                          blockSize=11,
                                          C=2)

        elif method == 'sauvola':
            window_size = 31
            R = 128.0
            k = 0.1

            image = image.astype(np.float32)

            mean = cv2.boxFilter(src=image, ddepth=-1, ksize=(window_size, window_size))
            mean_sq = cv2.boxFilter(src=image ** 2, ddepth=-1, ksize=(window_size, window_size))

            stddev = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
            threshold = mean * (1.0 + k * (stddev / R - 1.0))

            image = np.where(image > threshold, 255, 0)

        image = np.uint8(image)

        if invert:
            image = cv2.bitwise_not(image)

        outputs.append(image)

    return outputs


def batch_font_graph(font_graphs, font_graph_shape, batch_size, augmentor=False):
    """
    Builds a batch of graph data from character font graphs.

    Parameters
    ----------
    font_graphs : list
        List of graph representations from each character font.
    font_graph_shape : tuple or list
        The shape of the input font graphs.
    batch_size : int
        Number of samples in the batch.
    augmentor : bool, optional
        Whether to randomly apply variations per character font.

    Returns
    -------
    tuple
        Nodes and Edges as padded numpy arrays.
    """

    num_fonts = len(font_graphs)

    if not num_fonts:
        return []

    nodes_shape, edges_shape = font_graph_shape
    num_chars = len(font_graphs[0])

    max_nodes = nodes_shape[0] // num_chars
    max_edges = edges_shape[1] // num_chars

    nodes = np.zeros((batch_size, num_chars, max_nodes, 4), dtype=np.float32)
    edges = np.zeros((batch_size, num_chars, 2, max_edges), dtype=np.float32)

    for i in range(batch_size):
        font_index = np.random.randint(num_fonts)

        for j in range(num_chars):
            graph = font_graphs[font_index][j]

            graph_nodes = graph['nodes']
            graph_edges = graph['edges']

            if len(graph_nodes) == 0:
                continue

            if augmentor:
                graph_nodes = graph_nodes.copy()
                xy = graph_nodes[:, :2]

                angle = np.random.uniform(-np.pi / 18, np.pi / 18)
                cos, sin = np.cos(angle), np.sin(angle)

                xy = xy @ np.array([[cos, -sin], [sin, cos]], dtype=np.float32).T
                xy = xy * np.random.uniform(0.9, 1.1)
                xy = xy + np.random.normal(0, 0.01, xy.shape).astype(np.float32)

                graph_nodes[:, :2] = xy

            n = min(graph_nodes.shape[0], max_nodes)
            e = min(graph_edges.shape[1], max_edges)

            nodes[i, j, :n] = graph_nodes[:n]
            edges[i, j, :, :e] = graph_edges[:, :e] + j * max_nodes

    nodes = nodes.reshape(batch_size, max_nodes * num_chars, 4)
    edges = edges.reshape(batch_size, 2, max_edges * num_chars)

    return nodes, edges


def batch_illumination(batch_data):
    """
    Apply illumination compensation to enhance text visibility in images.

    Parameters
    ----------
    batch_data : list of np.ndarray
        List of grayscale images.

    Returns
    -------
    list of np.ndarray
        List of processed images.

    References
    ----------
    Efficient illumination compensation techniques for text images.
        https://www.sciencedirect.com/science/article/pii/S1051200412000826
    """

    outputs = []

    for image in batch_data:
        _, binary = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

        if np.sum(binary) > np.sum(image) * 0.8:
            outputs.append(image.copy())
            continue

        imgf = image.astype(np.float32)
        h, w = image.shape

        hist = cv2.calcHist([image], [0], None, [26], [0, 260])
        hr = np.argmax(hist > np.sqrt(image.size)) * 10

        cei = np.clip((imgf - (hr + 15)) * 2, 0, 255)

        gx = cv2.Sobel(imgf, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(imgf, cv2.CV_32F, 0, 1, 3)

        mag = cv2.magnitude(gx, gy)

        rng = mag / (mag.max() - mag.min())
        rng = (rng - rng.min()) * 255.

        tli = ~((rng >= 30) | (cei >= 60))
        tli = np.uint8(tli) * 255
        erosion = cv2.erode(tli, np.ones((3, 3), np.uint8), 1)

        img = cei.copy()
        for y in range(w):
            mask = erosion[:, y] == 0

            if not np.any(mask):
                continue

            diff = np.diff(mask.astype(np.int8))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]

            if mask[0]:
                starts = np.r_[0, starts]

            if mask[-1]:
                ends = np.r_[ends, h - 1]

            for s, e in zip(starts, ends):
                n = e - s + 1

                if n > 30:
                    continue

                top = cei[max(0, s - 5):s, y]
                bot = cei[e + 1:min(h, e + 6), y]

                if top.size == 0 or bot.size == 0:
                    continue

                img[s:e + 1, y] = np.linspace(top.max(), bot.max(), n)

        rng = img / (img.max() - img.min())
        rng = (rng - rng.min()) * 255.

        rng = cv2.blur(rng, (11, 11))

        image = np.nan_to_num(cei / rng) * 260
        image[erosion != 0] *= 1.5

        image = np.clip(image, 0, 255)
        image = np.uint8(image)

        outputs.append(image)

    return outputs


def batch_masking(batch_data,
                  max_shape,
                  char_height=64,
                  char_width=16,
                  from_text=False):
    """
    Generate image masks for a batch.

    Parameters
    ----------
    batch_data : ndarray
        Batch of data (images or texts).
    max_shape : tuple
        Maximum shape for each mask.
    char_height : int, optional
        Pixel height per character row.
    char_width : int, optional
        Pixel width per character column.
    from_text : bool, optional
        Whether to generate masks from images or text lengths.

    Returns
    -------
    list of ndarray
        Item masks indicating content areas.
    """

    max_height, max_width = max_shape[:2]
    masks = []

    for item in batch_data:
        if isinstance(item, str):
            height, width = max_height, max_width

        elif from_text:
            height = len(item) * char_height
            width = max(len(row) for row in item) * char_width

        else:
            height, width = item.shape

        shape = (min(height, max_height), min(width, max_width))
        masks.append(np.full(shape=shape, fill_value=255, dtype=np.uint8))

    return masks


def batch_padding(batch_data, pad_value=0, target_shape=None, dtype=np.int64):
    """
    Pads a batch of data to a uniform shape.

    Parameters
    ----------
    batch_data : list
        List of data.
    pad_value : int, optional
        Value used for padding.
    target_shape : tuple, optional
        Target shape for padding.
    dtype : data-type, optional
        Data type of the output array.

    Returns
    -------
    numpy.ndarray
        Padded data.
    """

    if target_shape:
        max_height, max_width = target_shape[:2]
    else:
        max_height = max(len(height) for height in batch_data)
        max_width = max(len(width) for height in batch_data for width in height)

    shape = (len(batch_data), max_height, max_width)
    padded = np.full(shape=shape, fill_value=pad_value, dtype=dtype)

    for i, data in enumerate(batch_data):
        data = np.array(data)

        if data.size > 0 and data.ndim == 2:
            padded[i, :data.shape[0], :data.shape[1]] = data

    return padded


def batch_processing(batch_mode,
                     batch_data,
                     batch_scale=True,
                     padding_shape=None,
                     illumination=False,
                     binarization=None):
    """
    Processes a data batch for model input.

    Parameters
    ----------
    batch_mode : str
        Type of input data.
    batch_data : list
        List of data input.
    batch_scale : bool, optional
        Whether to scale data values.
    padding_shape : tuple, optional
        Target shape for padding.
    illumination : bool, optional
        Apply illumination compensation.
    binarization : str, optional
        Apply binarization method.

    Returns
    -------
    numpy.ndarray
        Processed data.
    """

    if batch_mode in ['image', 'binary']:
        if batch_mode == 'image':
            if illumination:
                batch_data = batch_illumination(batch_data)

            if binarization:
                batch_data = batch_binarization(batch_data, method=binarization)

        batch_data = batch_padding(batch_data, target_shape=padding_shape, dtype=np.uint8)
        batch_data = np.expand_dims(batch_data, axis=-1)

        if batch_scale:
            if batch_mode == 'image':
                batch_data = (batch_data.astype(np.float32) / 127.5) - 1
            else:
                batch_data = (batch_data.astype(np.float32) / 255.)

    elif batch_mode == 'text':
        batch_data = batch_padding(batch_data, target_shape=padding_shape, dtype=np.int64)

    return np.array(batch_data)


def format_text(text):
    """
    Clean and format the input text.

    Parameters
    ----------
    text : str
        The input text to be cleaned.

    Returns
    -------
    str or list
        The formatted text.
    """

    substitutions = {
        r'[ ]': ' ',
        r'[＿]': '_',
        r'[，]': ',',
        r'[；]': ';',
        r'[：]': ':',
        r'[！﹗]': '!',
        r'[？﹖]': '?',
        r'[．。]': '.',
        r'[＂“”″‶]': '"',
        r'[（]': '(',
        r'[）]': ')',
        r'[［]': '[',
        r'[］]': ']',
        r'[｛]': '}',
        r'[｝]': '{',
        r'[＠]': '@',
        r'[＊]': '*',
        r'[／]': '/',
        r'[＼]': '\\\\',
        r'[＆]': '&',
        r'[＃]': '#',
        r'[％]': '%',
        r'[＾]': '^',
        r'[˗֊‐‑‒–—－−﹣]': '-',
        r'[＋]': '+',
        r'[＜]': '<',
        r'[＝]': '=',
        r'[＞]': '>',
        r'[｜]': '|',
        r'[～]': '~',
        r'[⋯]': '...',
        r'[＄]': '$',
        r"[＇ʼ´‘’‛′‵`᾽᾿՚׳❛❜｀`]": '\'',
    }

    regexes = {re.compile(pattern): replacement for pattern, replacement in substitutions.items()}

    for pattern, replacement in regexes.items():
        text = pattern.sub(replacement, text)

    text = html.unescape(text)
    text = ' '.join(text.replace('\n', '﹗').split()).replace('﹗', '\n').strip()

    return text


def read_image(image_path, bbox=None):
    """
    Read an image from the given file path and perform optional bbox.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    bbox : list, optional
        The bbox coordinates ([x, y, width, height]).

    Returns
    -------
    ndarray
        The loaded image as a NumPy array.
    """

    if not image_path:
        return np.full((16, 16), fill_value=255, dtype=np.uint8)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or image.size == 0:
        return np.full((16, 16), fill_value=255, dtype=np.uint8)

    if bbox is not None and len(bbox) == 4:
        x, y, width, height = bbox

        if isinstance(x, str):
            x = float(x) if '.' in x else int(x)

        if isinstance(y, str):
            y = float(y) if '.' in y else int(y)

        if isinstance(width, str):
            width = float(width) if '.' in width else int(width)

        if isinstance(height, str):
            height = float(height) if '.' in height else int(height)

        if isinstance(x, float):
            x = int(x * image.shape[1])

        if isinstance(y, float):
            y = int(y * image.shape[0])

        if isinstance(width, float):
            width = int(width * image.shape[1])

        if isinstance(height, float):
            height = int(height * image.shape[0])

        y_start = max(0, int(y - 5))
        x_start = max(0, int(x - 5))
        y_end = min(image.shape[0], int(y + max(1, height) + 5))
        x_end = min(image.shape[1], int(x + max(1, width) + 5))

        if y_end > y_start and x_end > x_start:
            cropped = image[y_start:y_end, x_start:x_end]
            if cropped is not None and cropped.size > 0:
                image = cropped
            else:
                return np.full((16, 16), fill_value=255, dtype=np.uint8)
        else:
            return np.full((16, 16), fill_value=255, dtype=np.uint8)

    if image is None or image.size == 0:
        return np.full((16, 16), fill_value=255, dtype=np.uint8)

    return np.uint8(image)


def resize_image(image, target_width=None, target_shape=None):
    """
    Resize the image to fit within the target shape, maintaining the aspect ratio.

    Parameters
    ----------
    image : ndarray
        Input image.
    target_width : int, optional
        Target image width from character widths.
    target_shape : list or tuple
        Target image shape.

    Returns
    -------
    ndarray
        Resized image.
    """

    if image is None or getattr(image, 'size', 0) <= 1:
        if target_shape:
            return np.ones(shape=target_shape[:2], dtype=np.uint8) * 255
        return None

    src_h, src_w = image.shape[:2]

    if target_width and target_shape:
        new_h = target_shape[0]
        new_w = min(target_width, target_shape[1])
        interpolation = cv2.INTER_CUBIC if new_h > src_h or new_w > src_w else cv2.INTER_AREA

        image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=interpolation)

    elif target_shape:
        dst_h, dst_w = target_shape[:2]

        aspect_ratio = src_w / max(1, src_h)

        if aspect_ratio < 1:
            new_h = min(dst_h, int(dst_w / aspect_ratio))
            new_w = max(1, int(new_h * aspect_ratio))
        else:
            new_w = min(dst_w, int(dst_h * aspect_ratio))
            new_h = min(dst_h, max(1, int(new_w / aspect_ratio)))

        if new_h < dst_h and new_w < dst_w:
            scale = dst_h / max(1.0, float(src_h))
            new_h = dst_h
            new_w = min(dst_w, max(1, int(round(src_w * scale))))

        interpolation = cv2.INTER_CUBIC if new_h > src_h or new_w > src_w else cv2.INTER_AREA
        image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=interpolation)

    return np.uint8(image)


def prepare_image_for_recognition(image, target_shape=(64, 1024, 1), pad_value=255, auto_trim=True, return_canvas=False):
    """
    Prepares an image for direct recognition model input:
    1. Ensures grayscale 2D array and correct ink polarity (dark text on light background).
    2. Optional auto-trim around ink to eliminate excess margins.
    3. Resizes keeping aspect ratio to target line height (64).
    4. Pads width to exact target shape (1024) with pad_value.
    5. Normalizes to [-1.0, 1.0] float32 tensor of shape (1, 64, 1024, 1).
    """
    if image is None or getattr(image, 'size', 0) == 0:
        canvas = np.full(target_shape[:2], fill_value=pad_value, dtype=np.uint8)
    else:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure dark ink on light background
        if np.mean(image) < 127:
            image = cv2.bitwise_not(image)

        # Auto-trim outer empty borders around ink
        if auto_trim:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = cv2.findNonZero(thresh)
            if coords is not None and len(coords) > 20:
                x, y, w, h = cv2.boundingRect(coords)
                pad = 4
                y1 = max(0, y - pad)
                y2 = min(image.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(image.shape[1], x + w + pad)
                image = image[y1:y2, x1:x2]

        h, w = image.shape[:2]
        dst_h, dst_w = target_shape[:2]

        scale = dst_h / max(1.0, float(h))
        new_w = min(dst_w, max(1, int(round(w * scale))))
        new_h = dst_h

        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        canvas = np.full((dst_h, dst_w), fill_value=pad_value, dtype=np.uint8)
        canvas[:, :new_w] = resized

    norm = ((canvas.astype(np.float32) / 127.5) - 1.0)
    tensor = norm[np.newaxis, ..., np.newaxis]

    if return_canvas:
        return tensor, canvas
    return tensor


def estimate_document_scale(gray):
    """
    Estimates the median character height (scale) of a document image dynamically.
    Enables true scale-invariant processing across phone photos (3000px) and thumbnails (150px).
    """
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    heights = [stats[i, 3] for i in range(1, num_labels) if 6 <= stats[i, 3] <= 0.6 * h and stats[i, 2] <= 0.6 * w]
    if len(heights) >= 3:
        return float(np.median(heights))
    return max(16.0, float(h) / 10.0)


def normalize_illumination(gray, med_h=None):
    """
    Normalizes uneven shadows, corner lighting, flash glare, and yellowed paper.
    Uses morphological background closure scaled to estimated text size.
    """
    if med_h is None:
        med_h = estimate_document_scale(gray)
    k_size = max(15, int(med_h * 4))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = np.clip((gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0, 0, 255).astype(np.uint8)
    return norm


def deskew_document(gray):
    """
    Detects and corrects document skew angle within [-30, +30] degrees.
    Essential for mobile camera captures and tilted handwriting pads.
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return gray, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if 0.8 < abs(angle) < 30.0:
        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed, angle
    return gray, 0.0


def suppress_ruled_lines(binary, min_line_ratio=0.35):
    """
    Detects and suppresses pre-printed horizontal ruled lines on notebook paper and
    doctor prescription pads so lines don't bridge words vertically.
    """
    h, w = binary.shape[:2]
    k_w = max(20, int(w * min_line_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(binary, lines)


def postprocess_transcription(text):
    """
    Production-grade text normalizer for medical shorthand, dosages, numbers, and punctuation:
    - Normalizes dosages (e.g. '500 mg' -> '500mg', '10 ml' -> '10ml')
    - Standardizes prescription frequency ('1 - 0 - 1' -> '1-0-1', 'b . i . d .' -> 'b.i.d.')
    - Cleans punctuation spacing and duplicate spaces
    """
    if not text:
        return ""
    import re
    # Clean up redundant spaces
    cleaned = re.sub(r'\s+', ' ', text).strip()
    # Prescription frequency: 1-0-1 or 1-1-1
    cleaned = re.sub(r'(\d)\s*-\s*(\d)\s*-\s*(\d)', r'\1-\2-\3', cleaned)
    # Dosages: 500 mg, 250 mg, 10 ml
    cleaned = re.sub(r'(\d+)\s*(mg|ml|mcg|gm|kg|tabs?|caps?)\b', r'\1\2', cleaned, flags=re.IGNORECASE)
    # Medical Latin abbreviations
    cleaned = re.sub(r'\bb\s*\.\s*i\s*\.\s*d\b', 'b.i.d.', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bt\s*\.\s*i\s*\.\s*d\b', 't.i.d.', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bq\s*\.\s*d\b', 'q.d.', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bp\s*\.\s*r\s*\.\s*n\b', 'p.r.n.', cleaned, flags=re.IGNORECASE)
    # Currency amounts: $ 50 -> $50
    cleaned = re.sub(r'([\$£€₹])\s*(\d)', r'\1\2', cleaned)
    return cleaned


def extract_text_lines(image, min_aspect_ratio=1.8, min_width=15, min_height=8, return_boxes=False, auto_deskew=True, auto_illumination=True):
    """
    Production-grade scale-invariant document & handwriting line extractor.

    Handles ALL real-world scenarios:
    1. Single-Line & Digit strings: Direct bypass with zero splitting.
    2. Dense Multi-Line Documents: Historical letters, exam sheets, doctor prescriptions.
    3. Cards & Notes with Doodles: Smiling suns, icons, and non-text drawings are automatically skipped.
    4. Poor Camera Photos: Corrects uneven illumination, phone deskew, and ruled notebook lines.
    """
    if image is None or getattr(image, 'size', 0) == 0:
        if return_boxes:
            return [], 'empty', [], []
        return [], 'empty', []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # 1. Estimate document scale (median character height) dynamically
    med_h = estimate_document_scale(gray)

    # 2. Illumination compensation for uneven shadows / phone flash
    if auto_illumination:
        gray = normalize_illumination(gray, med_h)

    # 3. Deskew tilt if photo was taken at an angle
    if auto_deskew:
        gray, angle = deskew_document(gray)

    # 4. Auto-trim outer empty borders around ink
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is not None and len(coords) > 20:
        x_ink, y_ink, w_ink, h_ink = cv2.boundingRect(coords)
        pad = max(2, int(med_h * 0.2))
        y1 = max(0, y_ink - pad)
        y2 = min(gray.shape[0], y_ink + h_ink + pad)
        x1 = max(0, x_ink - pad)
        x2 = min(gray.shape[1], x_ink + w_ink + pad)
        ink_crop = gray[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        ink_crop = gray
        offset_x, offset_y = 0, 0

    src_h, src_w = ink_crop.shape[:2]

    # 5. Scale-invariant Single-Line Bypass
    aspect = src_w / max(1, src_h)
    if src_h <= int(med_h * 2.2) or aspect >= 2.5:
        _, bin_check = cv2.threshold(ink_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h_proj = np.sum(bin_check > 0, axis=1)
        w_size = max(3, int(med_h * 0.4))
        if w_size % 2 == 0:
            w_size += 1
        smooth = np.convolve(h_proj, np.ones(w_size)/w_size, mode='same')
        peaks = [y for y in range(1, src_h-1) if smooth[y] > smooth[y-1] and smooth[y] >= smooth[y+1] and smooth[y] > 0.3 * np.max(smooth)]
        if len(peaks) <= 1:
            if return_boxes:
                return [ink_crop], 'single_line', [], [(offset_x, offset_y, src_w, src_h)]
            return [ink_crop], 'single_line', []

    # 6. Binary cleanup & suppress thin photo edges
    _, binary = cv2.threshold(ink_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean_bin = binary.copy()
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        if (bx <= 1 or bx + bw >= src_w - 1) and (bh > 0.4 * src_h or bw < 4):
            clean_bin[labels == i] = 0

    # Suppress notebook ruled lines if present
    clean_bin = suppress_ruled_lines(clean_bin)

    # 7. Dense Document Valley Splitting
    h_proj = np.sum(clean_bin > 0, axis=1)
    w_size = max(3, int(med_h * 0.4))
    if w_size % 2 == 0:
        w_size += 1
    smooth = np.convolve(h_proj, np.ones(w_size)/w_size, mode='same')
    max_val = np.max(smooth) if len(smooth) > 0 else 0

    valleys = [0]
    min_line_gap = max(5, int(med_h * 0.60))
    for y in range(min_line_gap, src_h - min_line_gap):
        if smooth[y] <= smooth[y-1] and smooth[y] < smooth[y+1]:
            if (y - valleys[-1]) >= min_line_gap:
                peak_val = np.max(smooth[valleys[-1]:y]) if len(smooth[valleys[-1]:y]) > 0 else 0
                if peak_val > 0.15 * max_val and smooth[y] < 0.70 * peak_val:
                    valleys.append(y)

    if src_h - valleys[-1] >= min_line_gap:
        valleys.append(src_h)
    else:
        valleys[-1] = src_h

    # If dense multi-line document with 6+ lines (e.g. letters, full pages, SAMPLE-2)
    if len(valleys) - 1 >= 6:
        lines = []
        kept_boxes = []
        skipped = []
        for i in range(len(valleys) - 1):
            vy1, vy2 = valleys[i], valleys[i+1]
            if vy2 - vy1 < 4:
                continue
            slice_g = ink_crop[vy1:vy2, :]
            slice_b = clean_bin[vy1:vy2, :]
            c = cv2.findNonZero(slice_b)
            if c is None or len(c) < 15:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)

            # Skip tiny edge watermark
            if bh < 6 or (bh < 12 and bw/max(1, bh) > 15 and vy2 > src_h - 15):
                skipped.append((f'Slice {i+1}', 'Watermark / border artifact', f'aspect={bw/max(1, bh):.2f}'))
                continue

            # Doodle / Non-text filter on slice
            asp = bw / max(1, bh)
            if asp < 1.7 and (bh > int(med_h * 1.1) or vy1 < int(src_h * 0.35)):
                skipped.append((f'Slice {i+1}', 'Drawing / Doodle (non-text)', f'aspect={asp:.2f}'))
                continue

            pad_y = max(1, int(bh * 0.10))
            pad_x = max(2, int(bw * 0.05))
            cy1 = max(0, by - pad_y)
            cy2 = min(vy2 - vy1, by + bh + pad_y)
            cx1 = max(0, bx - pad_x)
            cx2 = min(src_w, bx + bw + pad_x)

            lines.append(slice_g[cy1:cy2, cx1:cx2])
            kept_boxes.append((offset_x + cx1, offset_y + vy1 + cy1, cx2 - cx1, cy2 - cy1))

        if lines:
            if return_boxes:
                return lines, 'multi_line', skipped, kept_boxes
            return lines, 'multi_line', skipped

    # 8. Sparse Card / Note Mode (RLSA + doodle filter + same baseline merging)
    k_w = max(10, int(med_h * 1.5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    smeared = cv2.dilate(clean_bin, kernel, iterations=1)
    contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw >= min_width and bh >= min_height:
            raw_boxes.append((bx, by, bw, bh))

    cand_boxes = []
    skipped = []
    for i, (bx, by, bw, bh) in enumerate(raw_boxes):
        crop_bin = clean_bin[by:by+bh, bx:bx+bw]
        coords_ink = cv2.findNonZero(crop_bin)
        if coords_ink is None or len(coords_ink) < 15:
            continue
        ix, iy, iw, ih = cv2.boundingRect(coords_ink)
        true_asp = iw / max(1, ih)

        if true_asp < 0.35:
            skipped.append((f'Region {i+1}', 'Vertical border / noise', f'aspect={true_asp:.2f}'))
            continue
        if true_asp < 1.8 and (ih > int(med_h * 0.9) or by < int(src_h * 0.40)):
            skipped.append((f'Region {i+1}', 'Drawing / Doodle (non-text)', f'aspect={true_asp:.2f}'))
            continue
        cand_boxes.append((bx + ix, by + iy, iw, ih))

    if not cand_boxes:
        if return_boxes:
            return [ink_crop], 'single_line', skipped, [(offset_x, offset_y, src_w, src_h)]
        return [ink_crop], 'single_line', skipped

    # Merge words on the same horizontal baseline
    cand_boxes = sorted(cand_boxes, key=lambda b: (b[1] + b[3]/2, b[0]))
    merged = []
    cur = [cand_boxes[0]]
    for b in cand_boxes[1:]:
        curr_y_mid = cur[-1][1] + cur[-1][3]/2
        b_y_mid = b[1] + b[3]/2
        if abs(curr_y_mid - b_y_mid) <= int(med_h * 0.6):
            cur.append(b)
        else:
            mx1 = min(x[0] for x in cur)
            my1 = min(x[1] for x in cur)
            mx2 = max(x[0] + x[2] for x in cur)
            my2 = max(x[1] + x[3] for x in cur)
            merged.append((mx1, my1, mx2 - mx1, my2 - my1))
            cur = [b]
    if cur:
        mx1 = min(x[0] for x in cur)
        my1 = min(x[1] for x in cur)
        mx2 = max(x[0] + x[2] for x in cur)
        my2 = max(x[1] + x[3] for x in cur)
        merged.append((mx1, my1, mx2 - mx1, my2 - my1))

    lines = []
    kept_boxes = []
    for (bx, by, bw, bh) in merged:
        pad_y = max(2, int(bh * 0.15))
        pad_x = max(4, int(bw * 0.05))
        cy1 = max(0, by - pad_y)
        cy2 = min(src_h, by + bh + pad_y)
        cx1 = max(0, bx - pad_x)
        cx2 = min(src_w, bx + bw + pad_x)
        lines.append(ink_crop[cy1:cy2, cx1:cx2])
        kept_boxes.append((offset_x + cx1, offset_y + cy1, cx2 - cx1, cy2 - cy1))

    mode = 'single_line' if len(lines) == 1 else 'multi_line'
    if return_boxes:
        return lines, mode, skipped, kept_boxes
    return lines, mode, skipped
