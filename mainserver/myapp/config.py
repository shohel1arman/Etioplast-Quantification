import os
import re
import logging
from django.conf import settings
from dotenv import load_dotenv

# v1.1.0
# Load mainserver/.env explicitly (next to manage.py) so it works regardless of CWD.
load_dotenv(os.path.join(settings.BASE_DIR, '.env'))


def _env_float(name, default):
    val = os.getenv(name)
    if val in (None, ''):
        return default
    try:
        return float(val)
    except ValueError:
        return default


class Config:
    # Model weights (.pt). Set MODEL_PATH in mainserver/.env; relative paths are
    # resolved against the repository root (one level above manage.py).
    MODEL_PATH = os.getenv('MODEL_PATH', '')

    # Default output root (per-request folders are created under it)
    SAVE_DIR = os.path.join(settings.MEDIA_ROOT, 'detections')
    MEASURE_DECIMALS = 3

    # Detection / NMS
    CONFIDENCE_THRESHOLD = _env_float('CONFIDENCE_THRESHOLD', 0.5)
    # Optional per-class minimum confidence (internal class id -> conf). Classes not
    # listed use CONFIDENCE_THRESHOLD. YOLO is run at the lowest of these values and
    # each class is then filtered at its own threshold.
    CLASS_CONF_MIN = {}
    MIN_CONTOUR_AREA = 30
    IOU_THRESHOLD = 0.5

    # Containment helpers
    PARENT_DILATE_ITER = 2         # grow parent a bit so children on the edge fall inside
    PARENT_KERNEL = 3
    # Which parent region children are tested against:
    #   'mask' - the predicted etioplast mask (default, original behaviour)
    #   'hull' - its convex hull, so organelles sitting in a notch the model cut
    #            out of the envelope (often edge plastoglobules) still count.
    # Changing this changes counts; record the setting with your results.
    CONTAINMENT_MODE = os.getenv('CONTAINMENT_MODE', 'mask').strip().lower()

    # Overlap / inside-point minima (lenient)
    # key: class_id => min overlap fraction
    # 1: PLB, 2: Prothylakoid, 3: Plastoglobule, 4: Starch Grain
    CLASS_OVERLAP_MIN = {
        1: 0.45,
        2: 0.25,
        3: 0.45,
        4: 0.45,
    }
    CLASS_POINTS_INSIDE_MIN = {
        1: 0.50,
        2: 0.30,
        3: 0.45,
        4: 0.45,
    }

    # Etioplast completeness/shape
    BORDER_MARGIN = 2
    AR_MIN, AR_MAX = 0.55, 1.80
    EXTENT_MIN = 0.48
    RECT_FILL_MIN = 0.55
    POLY_EPS_FRAC = 0.03
    POLY_MIN, POLY_MAX = 3, 12

    # Drawing
    ALPHA_BLEND = 0.6
    LINE_THICKNESS = 3
    ELLIPSE_THICKNESS = 4
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    IMG_SIZE = int(_env_float('IMG_SIZE', 1024))

    # Default scale: pixels per micrometer (px/µm). There is no safe universal value,
    # so the default is "unset" (0) and requests must supply px_per_um unless
    # PX_PER_UM is set in .env.
    PX_PER_UM = _env_float('PX_PER_UM', 0.0)


os.makedirs(Config.SAVE_DIR, exist_ok=True)
if Config.CONTAINMENT_MODE not in ('mask', 'hull'):
    logging.getLogger(__name__).warning(
        f"Unknown CONTAINMENT_MODE '{Config.CONTAINMENT_MODE}', using 'mask'")
    Config.CONTAINMENT_MODE = 'mask'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_DEFINITIONS = {
    0: {'name': 'Etioplast', 'color': (255, 0, 0)},
    1: {'name': 'PLB', 'color': (0, 255, 0)},
    2: {'name': 'Prothylakoid', 'color': (0, 0, 255)},
    3: {'name': 'Plastoglobule', 'color': (255, 255, 0)},
    4: {'name': 'Starch Grain', 'color': (255, 0, 255)},
}

# Accepted spellings of model class names -> internal class id. The training
# notebook used "Prothylakid", so both spellings are mapped.
_NAME_ALIASES = {
    'etioplast': 0, 'etioplasts': 0,
    'plb': 1, 'plbs': 1, 'prolamellarbody': 1,
    'prothylakoid': 2, 'prothylakid': 2, 'prothylakoids': 2,
    'plastoglobule': 3, 'plastoglobules': 3, 'plastoglobuli': 3,
    'starchgrain': 4, 'starch': 4, 'starchgrains': 4,
}


def normalize_class_name(name: str) -> str:
    return re.sub(r'[^a-z]', '', str(name).lower())


def build_class_map(model_names) -> dict:
    """Map model class index -> internal class id using the model's own names.

    Unknown model classes are left out (their detections are ignored and logged),
    so a model trained with a different class order can't silently mislabel organelles.
    """
    if isinstance(model_names, (list, tuple)):
        model_names = dict(enumerate(model_names))
    mapping = {}
    for idx, name in model_names.items():
        internal = _NAME_ALIASES.get(normalize_class_name(name))
        if internal is None:
            logger.warning(f"Model class {idx} '{name}' is not a known organelle class; it will be ignored")
            continue
        mapping[int(idx)] = internal
    return mapping


class DetectedObject:
    def __init__(self, class_id, contour, mask, confidence, bbox, yolo_detection_idx):
        import cv2
        self.class_id = class_id
        self.contour = contour
        self.mask = mask
        self.confidence = float(confidence)
        self.yolo_idx = yolo_detection_idx
        self.bbox = bbox  # [x1,y1,x2,y2] in original-image pixels
        self.area = cv2.contourArea(contour)
        self.center = self._calculate_center()
        self.children = []
        self.parent = None
        self.is_valid = True
        self.object_id = None
        self.reject_reason = ''

    def _calculate_center(self):
        import cv2
        M = cv2.moments(self.contour)
        if M['m00'] != 0:
            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        return (int((self.bbox[0] + self.bbox[2]) / 2), int((self.bbox[1] + self.bbox[3]) / 2))

    def get_name(self):
        return CLASS_DEFINITIONS[self.class_id]['name']

    def get_color(self):
        return CLASS_DEFINITIONS[self.class_id]['color']
