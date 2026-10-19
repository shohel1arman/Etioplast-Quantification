# v1.1.0
import os
import threading
import logging
import cv2
import numpy as np

from .config import Config, CLASS_DEFINITIONS, DetectedObject, build_class_map

logger = logging.getLogger(__name__)

_YOLO = None
_CLASS_MAP = None
_LOCK = threading.RLock()


class ModelNotConfigured(RuntimeError):
    pass


def resolve_model_path() -> str:
    from django.conf import settings
    path = os.getenv('MODEL_PATH') or Config.MODEL_PATH
    if not path:
        raise ModelNotConfigured(
            "MODEL_PATH is not set. Put your trained YOLOv8-seg weights (.pt) somewhere on disk "
            "and set MODEL_PATH=<path> in mainserver/.env")
    if not os.path.isabs(path):
        repo_root = os.path.dirname(str(settings.BASE_DIR))
        for base in (repo_root, str(settings.BASE_DIR)):
            cand = os.path.join(base, path)
            if os.path.exists(cand):
                path = cand
                break
    if not os.path.exists(path):
        raise ModelNotConfigured(f"Model weights not found at: {path}")
    return path


def get_model():
    """Load the YOLO model once (thread-safe) and build the class-name mapping."""
    global _YOLO, _CLASS_MAP
    if _YOLO is None:
        with _LOCK:
            if _YOLO is None:
                from ultralytics import YOLO
                model_path = resolve_model_path()
                logger.info(f"Loading YOLO model: {model_path}")
                model = YOLO(model_path)
                if getattr(model, 'task', None) not in (None, 'segment'):
                    raise ModelNotConfigured(
                        f"Model task is '{model.task}', but this pipeline needs a segmentation (-seg) model")
                _CLASS_MAP = build_class_map(model.names)
                logger.info(f"Model classes: {model.names} -> internal ids {_CLASS_MAP}")
                if 0 not in _CLASS_MAP.values():
                    raise ModelNotConfigured(f"Model has no 'Etioplast' class: {model.names}")
                _YOLO = model
    return _YOLO


def get_class_map():
    get_model()
    return _CLASS_MAP


def model_info():
    model = get_model()
    return {'model_path': resolve_model_path(), 'task': model.task,
            'names': {int(k): v for k, v in model.names.items()},
            'class_map': {int(k): CLASS_DEFINITIONS[v]['name'] for k, v in _CLASS_MAP.items()}}


def _min_conf():
    vals = [Config.CONFIDENCE_THRESHOLD] + list(Config.CLASS_CONF_MIN.values())
    return min(vals)


def run_yolo_detection(self, image_path):
    logger.info(f"Running YOLO detection on: {os.path.basename(image_path)}")
    model = get_model()
    with _LOCK:  # Ultralytics predictors are not guaranteed thread-safe
        results = model.predict(
            source=image_path,
            imgsz=Config.IMG_SIZE,
            conf=_min_conf(),
            iou=Config.IOU_THRESHOLD,
            retina_masks=True,  # masks at original image resolution (no letterbox misalignment)
            save=False,
            verbose=False,
        )
    if not results or results[0].masks is None or len(results[0]) == 0:
        logger.warning("No detections found by YOLO")
        return None
    return results[0]


def extract_model_summary(self, yolo_result):
    """Raw (pre-hierarchy) per-class detection counts, after per-class confidence filtering."""
    if yolo_result.boxes is None:
        return False
    class_map = get_class_map()
    class_counts = {}
    for cid, conf in zip(yolo_result.boxes.cls.cpu().numpy().astype(int),
                         yolo_result.boxes.conf.cpu().numpy()):
        internal = class_map.get(int(cid))
        if internal is None or conf < Config.CLASS_CONF_MIN.get(internal, Config.CONFIDENCE_THRESHOLD):
            continue
        cname = CLASS_DEFINITIONS[internal]['name']
        class_counts[cname] = class_counts.get(cname, 0) + 1
    self.model_summary = class_counts
    self.stats['model_detections'] = class_counts
    logger.info(f"YOLO raw detections: {class_counts}")
    return True


def _instance_mask(yolo_result, i, orig_h, orig_w):
    """Binary uint8 mask (0/255) for detection i in original-image coordinates."""
    data = yolo_result.masks.data[i].cpu().numpy()
    if data.shape == (orig_h, orig_w):
        return (data > 0.5).astype(np.uint8) * 255
    # Fallback: rasterise the polygon, which Ultralytics already maps to original coords
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    poly = yolo_result.masks.xy[i]
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask


def convert_yolo_to_objects(yolo_result, image_path, min_contour_area=None):
    orig_h, orig_w = yolo_result.orig_shape[:2]
    image_shape = (orig_h, orig_w)
    class_map = get_class_map()

    class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
    confidences = yolo_result.boxes.conf.cpu().numpy()
    boxes = yolo_result.boxes.xyxy.cpu().numpy()  # already in original-image pixels

    if min_contour_area is None:
        min_contour_area = Config.MIN_CONTOUR_AREA

    detected = []
    counters = {}
    for i, (cid, conf, box) in enumerate(zip(class_ids, confidences, boxes)):
        internal = class_map.get(int(cid))
        if internal is None:
            continue
        if conf < Config.CLASS_CONF_MIN.get(internal, Config.CONFIDENCE_THRESHOLD):
            continue
        raw_mask = _instance_mask(yolo_result, i, orig_h, orig_w)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < min_contour_area:
            continue
        # Measure exactly the region that is drawn/exported: the filled main component.
        mask = np.zeros_like(raw_mask)
        cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

        bbox = [int(round(v)) for v in box]
        obj = DetectedObject(internal, main_contour, mask, float(conf), bbox, i)
        counters[internal] = counters.get(internal, 0) + 1
        obj.object_id = f"{CLASS_DEFINITIONS[internal]['name'].replace(' ', '_')}_{counters[internal]}"
        detected.append(obj)

    logger.info(f"Converted {len(detected)} YOLO detections to objects")
    return detected, image_shape
