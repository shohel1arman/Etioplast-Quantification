"""Offline batch analysis - no web server, no upload limit.

Examples (run from the folder that contains manage.py):

    python manage.py analyze_images "../Test Image" --px-per-um 344.8
    python manage.py analyze_images ../Dataset/test --px-per-um 344.8 --out ../results --recursive
    python manage.py analyze_images image.png --px-per-um 344.8 --conf 0.3

Writes, under --out:
    <image>/...                       per-image overlays, masks, measurements, diagnostics
    all_measurements.csv              one row per accepted etioplast across all images
    all_detections_diagnostics.csv    every raw detection with kept/rejected + reason
    run_summary.csv                   per-image counts (raw vs accepted) and errors
"""
import os
import csv
import datetime

from django.core.management.base import BaseCommand, CommandError

from myapp.config import Config, logger
from myapp.process import Process, is_image_file
from myapp.detection import ModelNotConfigured, model_info


def _collect(path, recursive):
    if os.path.isfile(path):
        return [path] if is_image_file(path) else []
    out = []
    if recursive:
        for root, _, files in os.walk(path):
            out += [os.path.join(root, f) for f in files if is_image_file(f)]
    else:
        out = [os.path.join(path, f) for f in os.listdir(path) if is_image_file(f)]
    return sorted(out)


def _concat_csv(paths, dest):
    header, rows = None, []
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        with open(p, newline='') as f:
            r = csv.reader(f)
            h = next(r, None)
            if h is None:
                continue
            header = header or h
            rows += list(r)
    if header:
        with open(dest, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    return len(rows)


class Command(BaseCommand):
    help = "Run etioplast detection + quantification on an image or a folder of images."

    def add_arguments(self, parser):
        parser.add_argument('input', help='Image file or folder of images')
        parser.add_argument('--px-per-um', type=float, default=None,
                            help='Pixels per micrometre for these images (falls back to PX_PER_UM in .env)')
        parser.add_argument('--out', default=None,
                            help='Output folder (default: media/detections/batch_<timestamp>)')
        parser.add_argument('--recursive', action='store_true', help='Include sub-folders')
        parser.add_argument('--conf', type=float, default=None,
                            help='Override confidence threshold for this run (e.g. 0.25 for diagnosis)')
        parser.add_argument('--imgsz', type=int, default=None, help='Override inference image size')
        parser.add_argument('--limit', type=int, default=None, help='Process at most N images')

    def handle(self, *args, **opts):
        src = os.path.abspath(opts['input'])
        if not os.path.exists(src):
            raise CommandError(f"Input not found: {src}")
        if opts['conf'] is not None:
            Config.CONFIDENCE_THRESHOLD = opts['conf']
        if opts['imgsz'] is not None:
            Config.IMG_SIZE = opts['imgsz']

        try:
            info = model_info()
        except ModelNotConfigured as e:
            raise CommandError(str(e))
        self.stdout.write(f"Model: {info['model_path']}")
        self.stdout.write(f"Classes used: {info['class_map']}")

        images = _collect(src, opts['recursive'])
        if opts['limit']:
            images = images[:opts['limit']]
        if not images:
            raise CommandError(f"No images (.png/.jpg/.tif/.bmp) found in {src}")

        out = opts['out'] or os.path.join(
            Config.SAVE_DIR, 'batch_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        out = os.path.abspath(out)
        os.makedirs(out, exist_ok=True)

        try:
            Process(px_per_um=opts['px_per_um'])  # validate scale before the loop
        except ValueError as e:
            raise CommandError(str(e))

        summary, meas, diags = [], [], []
        used = set()
        for i, path in enumerate(images, 1):
            base = os.path.splitext(os.path.basename(path))[0]
            name = base
            k = 1
            while name in used:  # same file name in different sub-folders
                name = f"{base}_{k}"
                k += 1
            used.add(name)
            self.stdout.write(f"[{i}/{len(images)}] {os.path.relpath(path, os.path.dirname(src))}")
            row = {'Image': os.path.basename(path), 'Path': path, 'Status': '', 'Raw_Detections': '',
                   'Valid_Etioplasts': 0, 'Rejected_Etioplasts': 0, 'Valid_Organelles': 0,
                   'Rejected_Organelles': 0, 'Error': ''}
            try:
                p = Process(px_per_um=opts['px_per_um'], save_dir=os.path.join(out, name))
                rep = p.process_image(path)
                if rep is None:
                    unreadable = (p.error or '').startswith('unreadable')
                    row.update(Status='error' if unreadable else 'no_detections', Error=p.error or '')
                else:
                    st = rep['processing_stats']
                    row.update(Status='ok',
                               Raw_Detections='; '.join(f"{k}={v}" for k, v in rep['model_summary'].items()),
                               Valid_Etioplasts=st['valid_etioplasts'],
                               Rejected_Etioplasts=st['rejected_etioplasts'],
                               Valid_Organelles=st['valid_organelles'],
                               Rejected_Organelles=st['rejected_organelles'])
                    meas.append(rep['outputs']['measurements_csv'])
                    diags.append(rep['outputs']['diagnostics_csv'])
            except Exception as e:  # keep going on a bad image
                logger.exception(f"Failed on {path}")
                row.update(Status='error', Error=str(e))
            summary.append(row)

        with open(os.path.join(out, 'run_summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        n_et = _concat_csv(meas, os.path.join(out, 'all_measurements.csv'))
        _concat_csv(diags, os.path.join(out, 'all_detections_diagnostics.csv'))

        ok = sum(r['Status'] == 'ok' for r in summary)
        none = sum(r['Status'] == 'no_detections' for r in summary)
        err = sum(r['Status'] == 'error' for r in summary)
        self.stdout.write(self.style.SUCCESS(
            f"\nDone: {len(summary)} images | {ok} analysed | {none} with no detections | {err} errors | "
            f"{n_et} accepted etioplasts\nResults: {out}"))
