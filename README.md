# Etioplast Analysis & Quantification Tool

**Automated instance segmentation, morphometric quantification, and AI-assisted interpretation of etioplast ultrastructures in electron microscopy images.**

This project is a **Django-based computer vision application** that uses a custom-trained **Ultralytics YOLOv8 segmentation model**, geometric image processing, and configurable spatial rules to analyze etioplasts and their internal structures. It aims to replace repetitive manual measurement with a reproducible pipeline for detecting biological structures, validating their spatial relationships, computing physical measurements, and presenting the results through visualizations, CSV exports, REST APIs, and an optional generative AI summary.

> **Project status:** Research software / active development. The pipeline currently faces missed detections, incomplete internal-organelle detection, and occasional images with no predictions. Quantification accuracy depends on model performance, annotation quality, image-specific scale calibration, and the validity of spatial filtering rules. No benchmark accuracy or biological validation claims are made here.

## Table of contents

- [Project objectives](#project-objectives)
- [Supported structures](#supported-structures)
- [How the analysis works](#how-the-analysis-works)
- [Morphometric measurements](#morphometric-measurements)
- [Features](#features)
- [Technology stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation and configuration](#installation-and-configuration)
- [Running the application](#running-the-application)
- [Using the REST API](#using-the-rest-api)
- [Output files](#output-files)
- [Dataset and further training](#dataset-and-further-training)
- [Known challenges and troubleshooting](#known-challenges-and-troubleshooting)
- [Limitations and research considerations](#limitations-and-research-considerations)
- [Security and reproducibility](#security-and-reproducibility)
- [Future improvements](#future-improvements)

## Project objectives

The research goal is to extract consistent, biologically meaningful measurements from high-resolution **electron microscopy (EM) images** of etioplasts. Specifically, the system is designed to:

1. Detect and segment etioplasts and five categories of ultrastructure (including the etioplast itself).
2. Reject incomplete or otherwise invalid parent etioplast detections.
3. Associate internal structures only with valid parent etioplasts.
4. Measure organelle areas, counts, lengths, and diameters in physical units.
5. Produce annotated visualizations and structured results suitable for biological analysis.
6. Generate optional natural-language summaries of quantitative findings using an external LLM API.
7. Support single-image and batch processing through Django and REST endpoints.

## Supported structures

The dataset contains **500 manually annotated high-resolution EM training images**. The five target classes are:

| Class ID* | Structure | Typical annotation | Analysis role |
| --- | --- | --- | --- |
| 0 | **Etioplast** | Polygon / instance mask | Parent region whose area is measured |
| 1 | **Prolamellar body (PLB)** | Polygon / instance mask | Required internal structure; area is measured |
| 2 | **Prothylakoid** | Polyline, converted to a thin mask for segmentation | Number and total length are measured |
| 3 | **Plastoglobule** | Small bounding box or mask | Number and estimated diameter are measured |
| 4 | **Starch grain** | Polygon / instance mask | Area can be measured |

\*This class-ID ordering is an **example project mapping**; it must match the actual trained model and dataset YAML. Never silently reassign IDs after training. Check `model.names` before inference.

**Annotation note:** Standard YOLOv8 *segmentation* training expects instance polygons rather than arbitrary mixed annotation geometries. Convert PLB/etioplast/starch polygons into segmentation labels, convert prothylakoid polylines to consistently defined narrow masks/polygons, and represent plastoglobule bounding boxes as suitable polygons or train a separate detector. Validate all converted labels visually before training.

## How the analysis works

```text
EM image(s)
    |
    v
Input validation and pixel-to-micrometer calibration
    |
    v
YOLOv8 inference: candidate instances across all five classes
    |
    v
Class-wise confidence filtering and mask extraction
    |
    v
Validate candidate etioplasts
    |-- Exclude parents touching image boundaries
    |-- Apply configurable shape/integrity checks
    `-- Require an associated, sufficiently contained PLB
    |
    v
Associate PLBs and secondary organelles with validated parents
    |-- Prothylakoids
    |-- Plastoglobules
    `-- Starch grains
    |
    v
Suppress unmatched or outside-parent detections
    |
    v
Area / count / length / diameter measurement
    |
    +--> CSV / structured data
    +--> Annotated overlays and binary masks
    `--> Optional LLM-based narrative interpretation
```

**Important distinction:** With a standard single-pass YOLOv8 model, the network predicts candidate objects first, and the *hierarchical rules are applied in post-processing*. The rules do not force YOLO to discover missing children. Truly sequential inference requires a separate second-stage approach, such as cropping validated etioplasts and rerunning a specialized detector.

### Hierarchical validation rules

**Rule 1 — Border exclusion.** Reject an etioplast if its mask touches or crosses the image edge, optionally with a configurable safety margin. This reduces partial-parent measurements.

**Rule 2 — Structural integrity.** Retain candidate parents that satisfy configured contour, aspect-ratio, size, or compactness rules. Shape thresholds are heuristics, **not universal biological definitions**: overly strict settings can reject genuine irregular etioplasts.

**Rule 3 — PLB dependency.** Require at least one detected PLB sufficiently contained within a candidate etioplast before that parent is accepted for analysis. If PLB inference fails, otherwise valid etioplasts may be lost.

**Rule 4 — Parent-child containment.** Associate each predicted PLB, prothylakoid, plastoglobule, or starch grain only with a validated parent, using mask intersection or an explicitly documented containment metric. A configurable overlap tolerance is often safer than exact 100% containment because segmentation boundaries are imperfect.

**Rule 5 — Background suppression.** Ignore isolated child predictions outside all validated parents for final quantification. Keep their raw predictions in diagnostic outputs where possible, so filtering errors are distinguishable from model misses.

If multiple valid etioplasts occur in one image, the analysis should give them distinct instance IDs and avoid double-counting shared child predictions. If the experiment requires **one selected etioplast per image**, configure an explicit selection rule and document it.

## Morphometric measurements

All physical measurements require accurate calibration: **`PX_PER_UM` = number of image pixels per micrometer**, preferably determined independently for each image or acquisition group.

| Measurement | Suggested computation | Unit |
| --- | --- | --- |
| Etioplast area | Parent segmentation-mask area / `PX_PER_UM²` | µm² |
| PLB area | Associated PLB mask area / `PX_PER_UM²` | µm² |
| Prothylakoid number | Count of associated, distinct prothylakoid instances | count |
| Total prothylakoid length | Sum of centerline / skeleton segment lengths / `PX_PER_UM` | µm |
| Plastoglobule number | Count of associated plastoglobule instances | count |
| Plastoglobule diameter | Estimated individual diameters; optionally report the mean | µm |
| Starch grain area | Associated starch-grain mask area / `PX_PER_UM²` | µm² |

For a plastoglobule mask with area `A_px`, an **equivalent-circle diameter** can be estimated as:

```text
diameter_um = 2 * sqrt(A_px / pi) / PX_PER_UM
```

This is an **estimate**, not a direct physical 3D diameter. Prothylakoid centerline lengths should account for diagonal steps (rather than simply counting all skeleton pixels as equal length), junctions, branches, and duplicate segments. Record the exact implementation alongside results.

**Calibration example:** `PX_PER_UM=344.8` means one micrometer spans 344.8 pixels. This is an illustrative value only; do not apply it to every EM image without verification. Wrong calibration affects lengths linearly and areas quadratically.

## Features

- **Automated detection and segmentation:** YOLOv8-based inference for five classes.
- **Hierarchy-aware filtering:** parent validation and parent-child assignment.
- **Morphometric analysis:** areas, counts, prothylakoid lengths, and plastoglobule diameter estimates.
- **Quality-control visuals:** annotated overlays, contours, optional bounding boxes, and binary instance/class masks.
- **Structured exports:** per-image and/or batch measurement tables for downstream statistics.
- **AI-assisted reports:** optional LLM-generated interpretation of quantified results; descriptive, not a substitute for biological validation.
- **Web application and API:** Django interface and routes intended for individual images and folder-based workflows.

## Technology stack

| Layer | Tools |
| --- | --- |
| Web / API | Python, Django; Django REST Framework if used by the repository |
| Object segmentation | Ultralytics YOLOv8 |
| Image operations | OpenCV, NumPy, scikit-image |
| Tabular exports | Python CSV / pandas if used by the repository |
| Generative AI | Configurable external LLM endpoint via API |
| Hardware | CPU supported; NVIDIA CUDA GPU recommended for faster inference and training |

## Prerequisites

- **Python 3.8+** as originally specified, but use a version actually supported by your pinned packages (a recent supported Python release is recommended).
- `pip` and a Python virtual environment.
- Trained **YOLOv8 segmentation weights** (`.pt`) compatible with your five-class label mapping.
- A valid microscopy calibration (`PX_PER_UM`), ideally per image.
- Optional: NVIDIA GPU with compatible CUDA/PyTorch installation.
- Optional: credentials for an LLM provider if generative summaries are enabled.

## Installation and configuration

Tested on Python 3.13 (CPU). Run all backend commands from `mainserver/` (the folder with `manage.py`).

```bash
git clone https://github.com/shohel1arman/Etioplast-Quantification.git
cd Etioplast-Quantification
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

For GPU inference, install a CUDA build of PyTorch first, following the PyTorch site for your OS/CUDA version.

### Model weights (required)

The `.pt` weights are not in Git (see `.gitignore`). Put your trained YOLOv8 **segmentation** model at, for example, `TrainedModelv2/best.pt`, then:

```bash
cd mainserver
cp .env.example .env      # Windows: copy .env.example .env
# edit .env -> MODEL_PATH=TrainedModelv2/best.pt   (absolute path, or relative to the repo root)
python manage.py migrate
python manage.py test myapp          # unit tests, no model needed
```

Class IDs are mapped **by the model's own class names** (`model.names`), not by index, so a 4-class model (no starch) or a different class order works. Recognised names include `Etioplast`, `PLB`, `Prothylakoid`/`Prothylakid`, `Plastoglobule`, `Starch Grain`; any other class is ignored with a log warning.

### Scale (`px_per_um`)

Pixels per micrometre must be supplied per request (or set `PX_PER_UM` in `.env`). There is no default: a missing or non-positive value returns HTTP 400 rather than producing wrong numbers.

## Running the application

**Backend** (port 8000):

```bash
cd mainserver
python manage.py runserver
curl http://127.0.0.1:8000/api/health/     # confirms the model loads and shows the class mapping
```

**Frontend** (port 5173):

```bash
cd Frontend
npm ci
npm run dev
```

Open http://localhost:5173, upload 1–5 images, enter px/µm, and run detection. If the backend is not on `127.0.0.1:8000`, create `Frontend/.env` with `VITE_API_BASE=http://<host>:8000`.

**Batch / offline (no web server, any number of images):**

```bash
cd mainserver
python manage.py analyze_images "../Test Image" --px-per-um 344.8 --out ../results
# options: --recursive  --conf 0.25 (diagnostic threshold)  --imgsz 640  --limit 20
```

## Using the REST API

| Method | Route | Form fields | Notes |
| --- | --- | --- | --- |
| GET | `/api/health/` | – | Model path, class names and mapping; 503 if the model is not configured |
| POST | `/api/analyze-file/` | `file`, `px_per_um` | Single image; returns `analysis`, `scale_used`, `output_image_url`, `output_urls`, `explanation`, `master_zip_url` |
| POST | `/api/analyze-folder/` | `files` (up to 5) or `file`, `px_per_um` | Returns `results[]` (one per image) and `master_zip_url` |

```bash
curl -X POST http://127.0.0.1:8000/api/analyze-file/ \
  -F "file=@/path/to/image.png" -F "px_per_um=344.8"
```

`explanation` is `null` unless `API_KEY`, `API_URL` and `LLM_MODEL` are set (any OpenAI-compatible chat-completions endpoint). Only the numeric summary is sent, never the image.

## Output files

Per image, in its own folder under `mainserver/media/detections/` (or `--out` for batch runs):

| File | Content |
| --- | --- |
| `<name>_model_first_detection.png` | Blended overlay of accepted structures |
| `<name>_overlay.png`, `<name>_contours_only.png` | Filled overlay / outlines only |
| `<name>_measurements.csv` | One row per accepted etioplast (areas µm², counts, prothylakoid length µm, plastoglobule diameter mean/SD µm, `PX_PER_UM`) |
| `<name>_detections_diagnostics.csv` | **Every raw detection**: class, confidence, area, bbox, kept/rejected, parent, rejection reason |
| `masks/` | Binary mask per accepted object and a combined mask |

Batch runs also write `all_measurements.csv`, `all_detections_diagnostics.csv` and `run_summary.csv` (raw vs accepted counts per image).

Use the diagnostics file to separate the two failure modes in the "Known challenges" section: a structure that **never appears** there was missed by the model; one listed as `rejected` was removed by a hierarchy rule.

### Measurement definitions (v1.1)

- Areas: pixel count of the filled main contour of each instance ÷ `px_per_um²`. Only the largest connected component of a predicted mask is used, so what is measured matches what is drawn.
- Prothylakoid length: skeleton centre-line, summing 1 px per orthogonal step and √2 px per diagonal step. (v1.0 counted skeleton pixels, which under-estimates diagonal segments by up to ~29%; lengths from v1.0 and v1.1 are not directly comparable.)
- Plastoglobule diameter: equivalent-circle diameter `2·sqrt(A/π)`, a 2D estimate.

## Dataset and further training

### Data source

**Google Drive training data and materials:**

[Etioplast dataset — Google Drive folder](https://drive.google.com/drive/folders/1XaGTpw5Pp90v5CoHgR4YDsl0juj2_x7x?usp=sharing)

The research dataset is described as **500 manually annotated EM training images** covering five classes. **The linked folder's exact files, download permissions, and current contents have not been independently verified** in this README. Review the folder, dataset licensing/permissions, image metadata, and annotations before use.

### Recommended dataset organization

After downloading and converting labels to a consistent YOLOv8 segmentation format, one possible layout is:

```text
datasets/
└── etioplast_yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    ├── labels/
    │   ├── train/
    │   ├── val/
    └── data.yaml
```

**Example `datasets/etioplast_yolo/data.yaml`:**

```yaml
path: datasets/etioplast_yolo
train: images/train
val: images/val
test: images/test
names:
  0: Etioplast
  1: PLB
  2: Prothylakoid
  3: Plastoglobule
  4: Starch Grain
```

Set `path` to an absolute path if your training working directory makes relative paths ambiguous. Ensure that the class order matches the actual annotations and model.

### Training a YOLOv8 segmentation model

If the annotations have been converted and checked, a **starting experiment** is:

```bash
yolo task=segment mode=train model=yolov8s-seg.pt data=datasets/etioplast_yolo/data.yaml epochs=150 imgsz=1024 batch=8
```

The parameters are **examples**, not an established best configuration. Adjust image size, batch size, augmentations, and model capacity based on microscopy detail, training/validation curves, and available GPU memory. Use the resulting `best.pt` as `MODEL_PATH` after evaluation.

Run validation on held-out data:

```bash
yolo task=segment mode=val model=runs/segment/train/weights/best.pt data=datasets/etioplast_yolo/data.yaml imgsz=1024
```

The actual run-directory name can differ. Do not train on the held-out test images or tune thresholds using the final test set. Where possible, split by biological specimen/acquisition session instead of randomly splitting highly related frames.

### Recommended training and evaluation checklist

1. **Annotation audit:** check missing labels, class IDs, thin prothylakoid geometry, tiny plastoglobules, and inconsistencies between annotators.
2. **Input quality:** preserve image resolution/contrast and investigate whether resizing makes small organelles disappear.
3. **Class-wise metrics:** inspect precision, recall, and mask quality separately for all five classes rather than relying only on aggregate mAP.
4. **Hierarchy diagnostics:** record raw detections **before** filtering and the reason every candidate was rejected.
5. **Parent-conditioned evaluation:** measure recall for PLB and other child classes **within ground-truth etioplasts**, distinct from full-image performance.
6. **Quantification validation:** compare areas, lengths, and diameters against manually reviewed reference measurements, not only segmentation metrics.
7. **Reproducibility:** save the model version, split manifest, calibration, preprocessing parameters, filtering thresholds, and training seed.

## Known challenges and troubleshooting

### 1. False negatives: obvious objects are missed

**Observed issue:** Human-recognizable structures receive no prediction or fall below the confidence threshold.

**Possible causes:** Too few representative examples, annotation inconsistency, domain shifts, EM contrast variation, low image resolution after resizing, class imbalance, and small-object difficulty.

**Investigations:** Lower confidence thresholds **for diagnostic visualization**, inspect raw class-wise predictions, evaluate different image sizes or overlapping tiles, add representative labeled data, and inspect false-negative patterns by imaging condition. Do not lower thresholds blindly for final quantitative results because false positives may rise.

### 2. Etioplast and PLB detected; children missing

**Observed issue:** The parent region and PLB are found, but prothylakoids or plastoglobules are not.

**Possible causes:** Tiny object size, limited contrast, polyline-to-mask representation, class imbalance, or rejection by overly strict containment.

**Investigations:** First distinguish **no raw child prediction** from **a predicted child filtered out**. Then consider higher-resolution crops of validated etioplasts, targeted small-object augmentation, dedicated second-stage models, and tolerance-aware mask containment.

### 3. No detections for an entire image

**Observed issue:** The model returns zero predictions across every class.

**Checklist:** Confirm the image loads; verify RGB/grayscale conversion, bit depth and normalization, model path and class mapping, compatible inference settings, and raw predictions at a diagnostic confidence threshold. Compare test-image acquisition conditions with training data, and check that overly aggressive parent rules are not making valid raw detections disappear from the final output.

### 4. Valid etioplast is rejected

**Possible causes:** Parent mask marginally touches a border, PLB was missed, shape heuristics are too restrictive, or child overlap thresholds are too high. Log individual rejection reasons and visualize them on the source image.

### 5. Incorrect physical measurements

**Possible causes:** Wrong `PX_PER_UM`, inconsistent resizing after calibration, mask holes, merged child instances, duplicate detections, or inaccurate skeletonization. Use image-specific scale metadata and manually verify representative results.

### 6. Django / model / LLM errors

- **`ModuleNotFoundError`:** activate the correct environment and install the repository's requirements.
- **Model not found:** verify `MODEL_PATH`, file permissions, and whether it points to segmentation weights.
- **CUDA unavailable or out of memory:** use CPU or a compatible PyTorch build; lower image size or batch size as needed.
- **Django misconfiguration:** inspect `settings.py`, `.env` loading, `ALLOWED_HOSTS`, migration status, and logs.
- **LLM request fails:** verify API URL, model ID, credentials, account access, and request schema; keep quantification usable even when the LLM is unavailable.

## Limitations and research considerations

- The pipeline measures **2D projections** from EM imagery; values should not be described as full 3D organelle volumes or true diameters.
- A **PLB-required parent rule** deliberately excludes PLB-negative/undetected etioplasts and can introduce selection bias. This should be reported in biological analyses.
- A prediction discarded by a hierarchical rule is not necessarily a network false negative; report both **raw-model** and **post-filter** metrics.
- Small-instance detection and long thin centerline segmentation may require task-specific architectures or loss functions.
- LLM output should be treated as **assisted narrative interpretation**, not experimental evidence or an independently verified diagnosis.
- No quantified model performance, validation study, supported endpoint contract, or deployment architecture was supplied; these should be added when available.

## Security and reproducibility

- Store API keys and Django secrets in environment variables; exclude `.env` from version control.
- Restrict file types, upload size, and filesystem access for user-supplied images/folders.
- Turn `DEBUG=False`, configure trusted hosts, and use a production WSGI/ASGI stack for deployment.
- Track exact model weights, package versions, dataset splits, calibration values, and processing rules with every export.
- Avoid transmitting confidential/unpublished source images to an LLM unless permitted by the data owner. Where possible, send only de-identified **quantitative summaries**.

## Future improvements

- **Two-stage parent-aware inference:** detect valid etioplasts, then analyze high-resolution crops for internal structures.
- **Small-object specialization:** improve plastoglobule detection and thin prothylakoid geometry handling.
- **Diagnostic rejection logs:** explicitly distinguish missing candidates from hierarchy-filtered candidates.
- **Per-image calibration:** automatic import of microscopy pixel-size metadata with manual override.
- **Expanded evaluation:** per-class recall/mAP, parent-child matching, area error, centerline-length error, and diameter error.
- **Batch reports:** cross-image QC, robust summary statistics, CSV/JSON export, and plotting.
- **Usability:** documented API schema, test cases, example images, reproducible setup, and optional containerized deployment.

---
