# v1.1.0
import os
import cv2
import numpy as np
from .config import Config, CLASS_DEFINITIONS, logger
from .geometry import mask_from_contour, points_inside_ratio, overlap_ratio, touches_border, is_square_like

def _parent_region(self, parent_obj):
    """Dilated parent mask used for containment, cached per parent and mode."""
    mode = Config.CONTAINMENT_MODE
    cache = getattr(parent_obj, '_region_cache', None)
    if cache and cache[0] == mode:
        return cache[1]
    contour = cv2.convexHull(parent_obj.contour) if mode == 'hull' else parent_obj.contour
    region = mask_from_contour(self.image_shape, contour, dilate_iters=Config.PARENT_DILATE_ITER)
    parent_obj._region_cache = (mode, region)
    return region


def strict_contains(self, child_obj, parent_obj):
    parent_mask = _parent_region(self, parent_obj)

    ov_min = Config.CLASS_OVERLAP_MIN.get(child_obj.class_id, 0.5)
    pt_min = Config.CLASS_POINTS_INSIDE_MIN.get(child_obj.class_id, 0.5)

    ov = overlap_ratio(child_obj.mask, parent_mask)
    if ov < ov_min:
        return False

    pts = points_inside_ratio(self.image_shape, child_obj.contour, parent_mask)
    if pts < pt_min:
        return False

    return True

def build_hierarchy(self):
    """Apply the parent/child rules. Every object ends with is_valid set and, if
    rejected, a human-readable reject_reason (used by the diagnostics export)."""
    logger.info("Building hierarchy with STRICT rules (valid-etioplast-only).")
    etioplasts = [o for o in self.detected_objects if o.class_id == 0]
    plbs = [o for o in self.detected_objects if o.class_id == 1]
    others = [o for o in self.detected_objects if o.class_id > 1]

    if not etioplasts:
        logger.warning("No Etioplasts detected by model!")
        for obj in self.detected_objects:
            obj.is_valid = False
            obj.reject_reason = 'no etioplast detected in image'
        self.stats['rejected_organelles'] = len(self.detected_objects)
        return

    etioplasts.sort(key=lambda x: x.area, reverse=True)
    valid_list = []
    for idx, et in enumerate(etioplasts):
        logger.info(f"Check Etioplast {idx+1}/{len(etioplasts)} (conf: {et.confidence:.2f}, area: {et.area:.0f})")
        reason = None
        plbs_inside = [p for p in plbs if p.parent is None and strict_contains(self, p, et)]
        if len(plbs_inside) == 0:
            reason, key = 'no PLB inside', 'reject_no_plb'
        elif touches_border(self.image_shape, et.contour):
            reason, key = 'touches image border (incomplete)', 'reject_incomplete'
        elif not is_square_like(et.contour):
            reason, key = 'failed shape check (not square-like)', 'reject_not_square_like'

        if reason:
            logger.warning(f"Rejected {et.object_id}: {reason}")
            et.is_valid = False
            et.reject_reason = reason
            self.stats['rejected_etioplasts'] += 1
            self.stats[key] += 1
            continue

        for p in plbs_inside:
            p.parent = et
            et.children.append(p)
        et.is_valid = True
        valid_list.append(et)

    for et in valid_list:
        for org in others:
            if org.parent is None and strict_contains(self, org, et):
                org.parent = et
                et.children.append(org)

    self.valid_etioplasts = valid_list
    self.stats['valid_etioplasts'] = len(valid_list)

    valid_children = 0
    rejected_children = 0
    for obj in self.detected_objects:
        if obj.class_id == 0:
            continue
        if obj.parent is not None and obj.parent.is_valid:
            obj.is_valid = True
            valid_children += 1
        else:
            obj.is_valid = False
            obj.parent = None
            obj.reject_reason = ('not inside any valid etioplast' if valid_list
                                 else 'no valid etioplast in image')
            rejected_children += 1

    self.stats['valid_organelles'] = valid_children
    self.stats['rejected_organelles'] = rejected_children

    logger.info(f"Valid Etioplasts: {self.stats['valid_etioplasts']}; "
                f"Valid organelles: {self.stats['valid_organelles']}; "
                f"Rejected organelles: {self.stats['rejected_organelles']}")

def draw_visualization_improved(self, img):
    overlay = img.copy()
    for et in self.valid_etioplasts:
        cv2.drawContours(overlay, [et.contour], -1, et.get_color(), Config.LINE_THICKNESS)
        label = f"Etioplast ({len([c for c in et.children if c.is_valid])})"
        pos = (et.center[0], et.center[1] - 10)
        cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE,
                    et.get_color(), Config.FONT_THICKNESS, cv2.LINE_AA)
        for org in et.children:
            if not org.is_valid:
                continue
            cv2.drawContours(overlay, [org.contour], -1, org.get_color(), cv2.FILLED)
            pos = (org.center[0], org.center[1] + 10)
            cv2.putText(overlay, org.get_name(), pos, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE,
                        org.get_color(), Config.FONT_THICKNESS, cv2.LINE_AA)
    return overlay

def save_outputs_mask(self, base_name):
    masks_dir = os.path.join(getattr(self, 'save_dir', Config.SAVE_DIR), 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    logger.info("💾 Saving individual masks...")

    combined_mask = np.zeros(self.image_shape, dtype=np.uint8)
    for idx, et in enumerate(self.valid_etioplasts, start=1):
        et_mask = np.zeros(self.image_shape, dtype=np.uint8)
        cv2.drawContours(et_mask, [et.contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(os.path.join(masks_dir, f"{base_name}_Etioplast_{idx}_mask.png"), et_mask)
        combined_mask = cv2.bitwise_or(combined_mask, et_mask)

        for j, org in enumerate([c for c in et.children if c.is_valid], start=1):
            om = np.zeros(self.image_shape, dtype=np.uint8)
            cv2.drawContours(om, [org.contour], -1, 255, thickness=cv2.FILLED)
            oname = org.get_name().replace(" ", "_")
            cv2.imwrite(os.path.join(masks_dir, f"{base_name}_Etioplast_{idx}_{oname}_{j}_mask.png"), om)
            combined_mask = cv2.bitwise_or(combined_mask, om)

    cv2.imwrite(os.path.join(masks_dir, f"{base_name}_combined_mask.png"), combined_mask)