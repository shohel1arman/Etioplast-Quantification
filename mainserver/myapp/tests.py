"""Tests that run without model weights: synthetic shapes with known geometry.

    cd mainserver && python manage.py test myapp
"""
import math
import os
import shutil
import tempfile

import cv2
import numpy as np
from django.test import SimpleTestCase, Client

from .config import Config, DetectedObject, build_class_map
from .geometry import MeasurementCalculator, _skeleton_length_px, summarize_for_api, write_measurements_csv
from .hierarchy_visualization import build_hierarchy

H, W = 1000, 1200


def make_obj(class_id, draw, idx=0, conf=0.9):
    """draw(mask) paints 255 into a blank mask; returns a DetectedObject like the detector does."""
    m = np.zeros((H, W), np.uint8)
    draw(m)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    o = DetectedObject(class_id, c, m, conf, [x, y, x + w, y + h], idx)
    o.object_id = f"c{class_id}_{idx}"
    return o


def ellipse(cx, cy, ax, ay):
    return lambda m: cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)


def circle(cx, cy, r):
    return lambda m: cv2.circle(m, (cx, cy), r, 255, -1)


def line(p1, p2, t=5):
    return lambda m: cv2.line(m, p1, p2, 255, t)


class FakeProc:
    def __init__(self, objs):
        self.detected_objects = objs
        self.image_shape = (H, W)
        self.valid_etioplasts = []
        self.stats = {k: 0 for k in ('processed_objects', 'valid_etioplasts', 'rejected_etioplasts',
                                     'valid_organelles', 'rejected_organelles', 'reject_no_plb',
                                     'reject_incomplete', 'reject_not_square_like')}


class GeometryTests(SimpleTestCase):
    def test_scale_must_be_positive(self):
        for bad in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                MeasurementCalculator(bad)
        self.assertAlmostEqual(MeasurementCalculator(100).px_area_to_um2(10000), 1.0)

    def test_skeleton_length_horizontal(self):
        m = np.zeros((50, 300), np.uint8)
        cv2.line(m, (50, 25), (249, 25), 255, 5)
        self.assertAlmostEqual(_skeleton_length_px(m), 200, delta=8)

    def test_skeleton_length_diagonal(self):
        m = np.zeros((300, 300), np.uint8)
        cv2.line(m, (50, 50), (250, 250), 255, 5)
        true_len = 200 * math.sqrt(2)  # ~282.8 px; pixel counting would give ~200
        self.assertAlmostEqual(_skeleton_length_px(m), true_len, delta=true_len * 0.05)

    def test_class_map_by_name_not_index(self):
        # different order + notebook spelling "Prothylakid" + an unknown class
        cmap = build_class_map({0: 'PLB', 1: 'Etioplast', 2: 'Prothylakid', 3: 'Plastoglobule', 4: 'Nucleus'})
        self.assertEqual(cmap, {0: 1, 1: 0, 2: 2, 3: 3})


class HierarchyTests(SimpleTestCase):
    def setUp(self):
        self.good = make_obj(0, ellipse(400, 500, 250, 200), 1)
        self.plb = make_obj(1, ellipse(400, 500, 80, 60), 2)
        self.pg_in = make_obj(3, circle(550, 450, 15), 3)
        self.pro_in = make_obj(2, line((250, 560), (330, 640)), 4)
        self.pg_out = make_obj(3, circle(1000, 150, 15), 5)
        self.no_plb = make_obj(0, ellipse(950, 600, 150, 140), 6)
        self.border = make_obj(0, ellipse(1150, 120, 150, 140), 7)  # crosses the right edge
        self.plb_border = make_obj(1, ellipse(1110, 120, 30, 30), 8)
        objs = [self.good, self.plb, self.pg_in, self.pro_in, self.pg_out,
                self.no_plb, self.border, self.plb_border]
        self.p = FakeProc(objs)
        build_hierarchy(self.p)

    def test_parent_rules(self):
        self.assertEqual(self.p.valid_etioplasts, [self.good])
        self.assertEqual(self.no_plb.reject_reason, 'no PLB inside')
        self.assertIn('border', self.border.reject_reason)
        self.assertEqual(self.p.stats['rejected_etioplasts'], 2)

    def test_children(self):
        for c in (self.plb, self.pg_in, self.pro_in):
            self.assertTrue(c.is_valid)
            self.assertIs(c.parent, self.good)
        for c in (self.pg_out, self.plb_border):
            self.assertFalse(c.is_valid)
            self.assertIsNone(c.parent)
        self.assertEqual(self.p.stats['valid_organelles'], 3)

    def test_measurements(self):
        px = 100.0  # 100 px per µm
        analysis, scale = summarize_for_api(self.p.valid_etioplasts, px)
        self.assertEqual(scale, '0.010000 µm/pixel')
        self.assertAlmostEqual(analysis['Etioplast']['total_area_um2'], math.pi * 250 * 200 / px ** 2, delta=0.1)
        plb_true = math.pi * 80 * 60 / px ** 2  # rasterised ellipse includes its boundary ring (~1-2%)
        self.assertAlmostEqual(analysis['PLB']['total_area_um2'], plb_true, delta=plb_true * 0.03)
        self.assertEqual(analysis['Plastoglobule']['count'], 1)
        self.assertAlmostEqual(analysis['Plastoglobule']['mean_diameter_um'], 0.30, delta=0.02)
        self.assertEqual(analysis['Prothylakoid']['count'], 1)
        self.assertAlmostEqual(analysis['Prothylakoid']['total_length_um'], 80 * math.sqrt(2) / px, delta=0.08)
        d = tempfile.mkdtemp()
        try:
            path = write_measurements_csv(self.p.valid_etioplasts, px, d, 'img')
            text = open(path).read()
            self.assertIn('PX_PER_UM', text.splitlines()[0])
            self.assertEqual(len(text.strip().splitlines()), 2)
        finally:
            shutil.rmtree(d)

    def test_no_etioplast_marks_everything_rejected(self):
        p = FakeProc([make_obj(1, ellipse(400, 500, 80, 60))])
        build_hierarchy(p)
        self.assertFalse(p.detected_objects[0].is_valid)


class ApiValidationTests(SimpleTestCase):
    def test_missing_scale_is_400(self):
        old = Config.PX_PER_UM
        Config.PX_PER_UM = 0
        try:
            ok, buf = cv2.imencode('.png', np.zeros((10, 10, 3), np.uint8))
            from django.core.files.uploadedfile import SimpleUploadedFile
            f = SimpleUploadedFile('a.png', buf.tobytes(), content_type='image/png')
            r = Client().post('/api/analyze-file/', {'file': f})
            self.assertEqual(r.status_code, 400)
            self.assertIn('px_per_um', r.json()['detail'])
        finally:
            Config.PX_PER_UM = old

    def test_get_not_allowed(self):
        self.assertEqual(Client().get('/api/analyze-file/').status_code, 405)


class ContainmentModeTests(SimpleTestCase):
    """Plastoglobule sitting in a notch the model cut out of the etioplast mask."""

    def run_mode(self, mode):
        def notched(m):
            cv2.ellipse(m, (500, 500), (250, 200), 0, 0, 360, 255, -1)
            cv2.circle(m, (745, 500), 30, 0, -1)  # notch on the right edge
        et = make_obj(0, notched, 1)
        plb = make_obj(1, ellipse(480, 500, 70, 50), 2)
        pg = make_obj(3, circle(738, 500, 14), 3)  # inside the notch
        old = Config.CONTAINMENT_MODE
        Config.CONTAINMENT_MODE = mode
        try:
            p = FakeProc([et, plb, pg])
            build_hierarchy(p)
        finally:
            Config.CONTAINMENT_MODE = old
        return pg

    def test_mask_mode_rejects_notch_child(self):
        self.assertFalse(self.run_mode('mask').is_valid)

    def test_hull_mode_accepts_notch_child(self):
        self.assertTrue(self.run_mode('hull').is_valid)
