# Road Damage Object Detection — Project Tasks

> **Project objective:** Build, compare, optimize, and deploy a YOLO-based object detection system that detects and localizes road damage in images.
>
> **Dataset:** [Road Damage dataset on Roboflow Universe](https://universe.roboflow.com/trafficsignssafeway/road-damage-l1ju7/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
>
> **Candidate model families:** YOLOv8, YOLO11, and YOLO26.
>
> **Deployment target:** A polished Streamlit application using the final selected model.

---

## Current Progress — Local Training Migration

- [x] Colab-only paths and workflow assumptions have been ported into local configs and scripts.
- [x] Secret handling is environment-based; no API key is stored in tracked files.
- [x] Local dataset download, conversion, validation, EDA, baseline training, tuning, evaluation, benchmarking, and export scripts have been added.
- [x] Dependencies are installed in `.venv` with `pip install -e .[dev]`.
- [x] Syntax verification passed with `.venv\Scripts\python.exe -m compileall -q src scripts tests`.
- [x] Ruff passed with `.venv\Scripts\python.exe -m ruff check .`.
- [x] Unit tests passed with `.venv\Scripts\python.exe -m pytest`.
- [x] Ultralytics settings are redirected to `.ultralytics/` inside the project.
- [x] Environment capture works and records CUDA-enabled Torch on the RTX 3050 Ti Laptop GPU.
- [x] Dataset download and validation gate passed for Roboflow version 1.
- [x] YOLOv8s baseline trained and validated locally; best validation mAP50-95 is `0.23420695689130214`.
- [ ] YOLO11s local run is incomplete/invalid for fair comparison: only 3 epochs finished, though `best.pt`, `last.pt`, and `results.csv` exist.
- [x] Roboflow project `roadddd-9ducw` version 2 has a finished hosted `YOLO11n` model with model-eval validation mAP50-95 `0.24429269006065163`.
- [x] Roboflow project `roadddd-9ducw` version 2 has a finished hosted `YOLO26s` model with model-eval validation mAP50-95 `0.22838667711885213`.
- [x] Current best available metric model remains hosted Roboflow `YOLO11n` after comparing YOLOv8, YOLO11, and YOLO26 by validation mAP50-95.
- [x] Active Streamlit deployment is temporarily switched to local `YOLOv8s` because its score is close and Roboflow serverless inference is blocked by credits.
- [ ] Hosted Roboflow smoke inference is currently blocked because serverless inference returns HTTP 402 `credit_cap_exceeded`.
- [x] Local `.pt` downloads are not available for the hosted Roboflow fine-tunes with the current API permissions.
- [x] Streamlit app shell supports local Ultralytics and hosted Roboflow inference and is configured for local YOLOv8s.
- [x] Streamlit UI has been refreshed into a dark professional inspection-console theme, with desktop and narrow/mobile screenshots updated.
- [x] Local YOLOv8s production-smoke inference passed on one validation image.
- [x] Annotation rendering, detection filtering, CSV/JSON export helpers, and related unit tests are implemented.
- [ ] Local YOLO26s baseline has not been started; only hosted Roboflow YOLO26s metrics have been imported.
- [x] Active local YOLOv8s fallback has been evaluated on the untouched local test split, benchmarked, exported to ONNX, and copied to production export paths.
- [ ] Fair same-size YOLO11s/YOLO26s baseline training, tuning, and hosted-inference reactivation remain open.

---

## 1. Definition of Done

The project is complete only when all of the following are satisfied:

- [ ] The dataset has been downloaded, versioned, validated, and documented.
- [ ] Dataset leakage, duplicate images, corrupt files, invalid labels, and severe class imbalance have been investigated.
- [ ] At least three YOLO versions have been trained under a fair baseline protocol.
- [ ] YOLOv8, YOLO11, and YOLO26 have been compared using the same dataset split, image size, evaluation code, and benchmark hardware.
- [ ] Hyperparameter tuning has been performed for every selected YOLO family under a documented compute budget.
- [ ] Final candidates have been retrained from their best configurations using multiple random seeds where resources allow.
- [ ] Models have been evaluated on an untouched test set.
- [ ] The comparison includes:
  - [ ] Precision
  - [ ] Recall
  - [ ] F1-score
  - [ ] mAP@0.50
  - [ ] mAP@0.50:0.95
  - [ ] Per-class AP
  - [ ] IoU analysis
  - [ ] Inference latency and FPS
  - [ ] Parameter count
  - [ ] Model file size
  - [ ] CPU/GPU memory consumption
  - [ ] Deployment/export efficiency
- [ ] Results include quantitative comparison, qualitative examples, error analysis, and limitations.
- [ ] The final model selection is justified using accuracy, speed, size, and deployment constraints—not mAP alone.
- [ ] A user-friendly Streamlit app supports image upload and camera capture.
- [ ] The deployed app displays detections, confidence scores, damage counts, and inference time.
- [ ] Unit, integration, and smoke tests pass.
- [ ] The repository contains reproducible instructions, experiment records, a model card, and a final report.

---

## 2. Project Rules and Experimental Controls

These rules apply to every experiment unless a task explicitly states otherwise.

- [ ] Use one frozen train/validation/test split for all model families.
- [ ] Never select hyperparameters or confidence thresholds using the test set.
- [ ] Use pretrained detection weights rather than training from scratch unless a separate ablation is planned.
- [ ] Compare equivalent model-size tiers first:
  - Primary comparison: `yolov8s`, `yolo11s`, and `yolo26s`.
  - Resource-constrained fallback: use the `n` variant for all three families.
- [ ] Do not compare a nano model from one family against a medium or large model from another and call it a fair architectural comparison.
- [ ] Keep image size, epochs, patience, augmentations, seed, optimizer policy, and evaluation thresholds identical during the baseline comparison.
- [ ] Record every intentional difference between model families.
- [ ] Run speed benchmarks on the same machine, runtime, precision, batch size, and input resolution.
- [ ] Fix and record random seeds.
- [ ] Store configuration in YAML files rather than hard-coding experiment settings.
- [ ] Save raw metrics and predictions so charts can be regenerated without retraining.
- [ ] Use Git for source code, but do not commit large datasets, checkpoints, API keys, secrets, or local experiment caches.

---

## 3. Phase 0 — Repository and Environment Setup

### RD-001 — Create the repository structure

- [ ] Create the initial project structure:

```text
road-damage-detection/
├── app/
│   ├── app.py
│   ├── components/
│   ├── services/
│   ├── assets/
│   └── styles/
├── configs/
│   ├── data/
│   ├── training/
│   ├── tuning/
│   └── deployment/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── reports/
├── models/
│   ├── checkpoints/
│   └── exports/
├── notebooks/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md
├── scripts/
│   ├── download_data.py
│   ├── validate_data.py
│   ├── analyze_data.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── export_model.py
├── src/
│   └── road_damage/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── .github/workflows/
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── README.md
├── TASKS.md
├── MODEL_CARD.md
└── LICENSE
```

**Acceptance criteria**

- [x] Project modules can be imported without modifying `sys.path`.
- [x] Generated artifacts are excluded through `.gitignore`.
- [x] Paths are configurable and work on Windows and Linux.

### RD-002 — Create a reproducible Python environment

- [x] Select and document the supported Python version.
- [x] Install and pin direct dependencies, including:
  - `ultralytics`
  - `torch` and `torchvision`
  - `opencv-python-headless`
  - `numpy`
  - `pandas`
  - `Pillow`
  - `matplotlib`
  - `plotly`
  - `scikit-learn`
  - `pyyaml`
  - `streamlit`
  - `optuna` or the chosen tuning framework
  - `pytest`
  - `ruff`
- [x] Record CUDA, cuDNN, GPU driver, PyTorch, Ultralytics, and operating-system versions.
- [x] Add `.env.example` for optional Roboflow credentials.
- [x] Confirm that secrets are loaded from environment variables and never committed.
- [x] Add a command that prints the full environment and hardware summary.

**Acceptance criteria**

- [x] A fresh environment can install the project from the documented commands.
- [x] `python -c "import torch; print(torch.cuda.is_available())"` is documented and tested.
- [x] CPU-only execution fails gracefully or uses a supported fallback.

### RD-003 — Configure code quality and continuous integration

- [x] Configure `ruff` for linting and formatting.
- [x] Configure `pytest`.
- [x] Add type hints to reusable modules.
- [ ] Add pre-commit hooks.
- [ ] Add a GitHub Actions workflow that:
  - installs dependencies;
  - runs linting;
  - runs unit tests;
  - runs a lightweight inference smoke test without downloading the full dataset.

---

## 4. Phase 1 — Dataset Acquisition, Governance, and Audit

### RD-010 — Download and version the dataset

- [ ] Download/export the dataset in Ultralytics YOLO object-detection format.
- [ ] Preserve the original dataset version separately from processed data.
- [ ] Save the dataset version, download date, source, class mapping, and preprocessing settings.
- [x] Store the Roboflow source URL in the dataset documentation.
- [ ] Verify and document the dataset license and attribution requirements before redistribution.
- [x] Do not commit the full dataset to Git unless its license and repository size make this appropriate.
- [ ] Create or validate `data.yaml` with relative, portable paths.

**Expected source labels to verify**

- `crack`
- `pothole`
- `block`
- `edge`
- `alligator`
- `custom`

> The `custom` label is ambiguous and must not be accepted blindly. Its semantic meaning and annotation consistency must be established during the audit.

### RD-011 — Validate images and annotations

- [ ] Count images in each split.
- [ ] Count objects per class and per split.
- [ ] Detect unreadable, truncated, or zero-byte image files.
- [ ] Verify every annotation has:
  - a valid class ID;
  - five YOLO fields;
  - finite numeric values;
  - normalized coordinates in `[0, 1]`;
  - positive width and height;
  - a bounding box that remains inside the image after conversion.
- [ ] Find images with missing labels.
- [ ] Distinguish intentionally negative images from annotation omissions.
- [ ] Detect duplicate and near-duplicate images using cryptographic and perceptual hashes.
- [ ] Check for duplicate or near-identical frames across train, validation, and test sets.
- [ ] Check whether images came from videos or bursts that require group-aware splitting.
- [ ] Generate a machine-readable validation report.
- [ ] Fail the validation script when critical defects exceed an agreed threshold.

**Acceptance criteria**

- [ ] No corrupt images remain in the processed dataset.
- [ ] No invalid bounding boxes remain.
- [ ] No confirmed duplicate appears across dataset splits.
- [ ] Every class ID maps to exactly one documented class name.

### RD-012 — Audit label quality

- [ ] Review a stratified sample from every class and split.
- [ ] Visualize random annotations and difficult examples.
- [ ] Review all or a substantial sample of the `custom` class.
- [ ] Decide whether `custom` should be:
  - renamed;
  - merged with another class;
  - relabeled;
  - excluded;
  - retained with a precise definition.
- [ ] Search for class overlap or inconsistent definitions such as:
  - `crack` versus `alligator`;
  - `edge` versus road boundary artifacts;
  - `block` versus repaired or patched surfaces.
- [ ] Identify boxes that are too loose, too tight, duplicated, or missing.
- [ ] Record annotation issues in `data/reports/label_audit.csv`.
- [ ] Correct labels only through a traceable process.
- [ ] Version the cleaned dataset separately.

**Acceptance criteria**

- [ ] Every retained class has a written visual definition.
- [ ] At least three representative positive examples and common confusions are documented per class.
- [ ] The final class list is approved before training begins.

### RD-013 — Produce dataset exploratory analysis

- [ ] Plot class frequency by split.
- [ ] Plot objects per image.
- [ ] Plot image width, height, and aspect ratio.
- [ ] Plot normalized bounding-box width, height, area, and aspect ratio by class.
- [ ] Measure the percentage of small, medium, and large boxes using a documented rule.
- [ ] Inspect class co-occurrence.
- [ ] Identify unusually dense images and extreme aspect ratios.
- [ ] Examine lighting, weather, motion blur, shadows, camera angle, occlusion, and road-surface variation.
- [ ] Document class imbalance and small-object risks.
- [ ] Save all plots under `reports/figures/data/`.

### RD-014 — Freeze the dataset split

- [ ] Preserve the original split if it is valid, representative, and leakage-free.
- [ ] Otherwise create a deterministic, stratified or group-aware split.
- [ ] Ensure rare classes are represented in validation and test sets.
- [ ] Save split manifests containing image IDs and hashes.
- [ ] Make the test set read-only for the duration of model development.
- [ ] Record the split seed and splitting strategy.

---

## 5. Phase 2 — Preprocessing and Augmentation Strategy

### RD-020 — Establish preprocessing

- [ ] Convert input images consistently to RGB.
- [ ] Respect EXIF orientation.
- [ ] Preserve aspect ratio using the detector's standard letterboxing behavior.
- [ ] Use `640 × 640` as the baseline input size.
- [ ] Add an optional higher-resolution experiment, such as `960 × 960`, only if EDA shows that small damage regions are a major limitation.
- [ ] Avoid image enhancement that cannot be reproduced during deployment.
- [ ] Document every preprocessing operation.

### RD-021 — Define realistic training augmentation

- [ ] Establish a baseline using the default or a fixed shared augmentation policy.
- [ ] Consider road-relevant augmentation ranges for:
  - brightness and contrast variation;
  - HSV changes;
  - shadows and illumination shifts;
  - mild blur and motion blur;
  - JPEG compression;
  - scaling and translation;
  - mild perspective changes;
  - horizontal flipping.
- [ ] Do not use vertical flipping unless the use case justifies upside-down roads.
- [ ] Review Mosaic, MixUp, and Copy-Paste effects rather than enabling them blindly.
- [ ] Disable or reduce aggressive augmentation during final epochs when supported.
- [ ] Confirm that augmentation does not create impossible road geometry or destroy thin cracks.
- [ ] Save before/after augmentation samples for review.

---

## 6. Phase 3 — Baseline Training

### RD-030 — Create a shared baseline configuration

- [x] Create one baseline YAML configuration shared by all candidate models.
- [x] Include:
  - dataset YAML path;
  - image size;
  - epoch limit;
  - early-stopping patience;
  - batch-size policy;
  - optimizer policy;
  - initial learning rate;
  - weight decay;
  - warm-up settings;
  - augmentation values;
  - random seed;
  - device;
  - workers;
  - experiment output path.
- [x] Prefer automatic batch sizing only when it produces a stable, recorded value.
- [x] Save the resolved training arguments with every run.
- [x] Enable checkpointing for `best` and `last`.
- [ ] Log training duration and peak GPU memory.

### RD-031 — Train YOLOv8 baseline

- [x] Train the selected YOLOv8 size tier using pretrained detection weights.
- [x] Save:
  - training curves;
  - validation predictions;
  - confusion matrix;
  - precision-recall curves;
  - per-class metrics;
  - checkpoint metadata;
  - training time and resource usage.
- [x] YOLOv8 artifact health check passed for checkpoint, curves, validation predictions, confusion matrix, PR curves, per-class metrics, and checkpoint metadata.
- [ ] Peak GPU memory/resource usage is not captured for the YOLOv8 training run.
- [x] Register the run in the experiment table.

### RD-032 — Train YOLO11 baseline

- [ ] Train the equivalent YOLO11 size tier with the same baseline protocol.
- [x] Evaluate the discovered incomplete local YOLO11s checkpoint and mark it invalid for fair comparison.
- [x] Import Roboflow-hosted YOLO11n metrics from project `roadddd-9ducw` version 2.
- [x] Register YOLO11n Roboflow model-eval metrics in `reports/final_comparison/model_comparison_current.csv`.
- [ ] Verify current Roboflow-hosted YOLO11n inference after the workspace credit cap is fixed.
- [x] Downloadable YOLO11 weights are not available through the current Roboflow API permissions.
- [ ] Save the same complete artifacts generated for YOLOv8 from a valid YOLO11s baseline run.
- [x] Register the run in the experiment table.

### RD-033 — Train YOLO26 baseline

- [ ] Train the equivalent YOLO26 size tier with the same baseline protocol.
- [x] Import Roboflow-hosted YOLO26s metrics from project `roadddd-9ducw` version 2.
- [x] Register YOLO26s Roboflow model-eval metrics in `reports/final_comparison/model_comparison_current.csv`.
- [ ] Save the same artifacts generated for YOLOv8 and YOLO11.
- [ ] Record any architecture-specific configuration that cannot be made identical.
- [x] Register the hosted run in the experiment table.

### RD-034 — Verify baseline fairness

- [ ] Confirm that all three baseline runs used the same:
  - split manifest;
  - image size;
  - epoch ceiling;
  - early-stopping policy;
  - augmentation policy;
  - batch size or batch-size rule;
  - seed;
  - hardware;
  - precision mode;
  - evaluation script.
- [ ] Mark any violation in the comparison table.
- [ ] Repeat invalid runs before drawing conclusions.

---

## 7. Phase 4 — Hyperparameter Tuning

### RD-040 — Define the tuning protocol

- [x] Choose and document the tuning engine:
  - Ultralytics tuning mode; or
  - Optuna with pruning and persistent storage.
- [x] Set an equal compute budget for each YOLO family.
- [x] Use validation `mAP@0.50:0.95` as the primary optimization metric.
- [ ] Track inference latency as a secondary deployment metric.
- [x] Define trial count, epoch budget, pruning policy, and timeout before tuning.
- [x] Use the same search-space philosophy for all models.
- [x] Do not tune on the test set.
- [x] Store every trial, including failed and pruned trials.

### RD-041 — Define the search space

Tune shared parameters where supported:

- [x] Initial learning rate.
- [x] Final learning-rate factor.
- [ ] Momentum or optimizer-specific equivalent.
- [x] Weight decay.
- [ ] Warm-up epochs and momentum.
- [ ] Batch size or gradient accumulation.
- [ ] Image size.
- [x] Box loss weight.
- [x] Classification loss weight.
- [ ] Model-supported localization/head loss parameters.
- [x] HSV augmentation.
- [x] Translation.
- [x] Scale.
- [x] Shear and perspective.
- [x] Horizontal-flip probability.
- [x] Mosaic.
- [x] MixUp.
- [x] Close-Mosaic schedule.

**Guardrails**

- [x] Do not force an unsupported parameter onto a model family.
- [ ] Keep model-specific search spaces documented separately.
- [ ] Reject configurations that cause out-of-memory errors repeatedly.
- [ ] Reject augmentation combinations that produce visibly invalid samples.

### RD-042 — Tune YOLOv8

- [ ] Run the planned YOLOv8 tuning study.
- [ ] Save the best configuration and top trials.
- [ ] Plot optimization history and parameter importance.
- [ ] Retrain the best configuration from a clean initialization.
- [ ] Validate that improvement is not due to a single lucky run.

### RD-043 — Tune YOLO11

- [ ] Run the planned YOLO11 tuning study using the same compute budget.
- [ ] Save the best configuration and top trials.
- [ ] Plot optimization history and parameter importance.
- [ ] Retrain the best configuration from a clean initialization.
- [ ] Validate that improvement is reproducible.

### RD-044 — Tune YOLO26

- [ ] Run the planned YOLO26 tuning study using the same compute budget.
- [ ] Save the best configuration and top trials.
- [ ] Plot optimization history and parameter importance.
- [ ] Retrain the best configuration from a clean initialization.
- [ ] Validate that improvement is reproducible.

### RD-045 — Multi-seed final training

- [ ] Retrain each tuned finalist using at least three seeds when compute permits.
- [ ] Report mean and standard deviation for the primary metrics.
- [ ] Use the same final epoch-selection rule for each family.
- [ ] Select one checkpoint per family without inspecting the test set.
- [ ] Document when only one seed is feasible and treat conclusions as less certain.

---

## 8. Phase 5 — Evaluation and Error Analysis

### RD-050 — Set the evaluation protocol

- [ ] Choose the confidence threshold using validation data.
- [ ] Choose any NMS IoU threshold using validation data where applicable.
- [ ] Freeze thresholds before final test evaluation.
- [x] Run evaluation at batch size 1 and the deployment image size for the active local YOLOv8s fallback.
- [x] Save image-level predictions in reusable JSONL format for local checkpoints.
- [ ] Report macro, micro, and per-class metrics where meaningful.

### RD-051 — Calculate required accuracy metrics

For each baseline and tuned final model, calculate:

- [x] Precision for active local YOLOv8s fallback.
- [x] Recall for active local YOLOv8s fallback.
- [x] F1-score for active local YOLOv8s fallback.
- [x] `mAP@0.50` for active local YOLOv8s fallback.
- [x] `mAP@0.50:0.95` for active local YOLOv8s fallback.
- [x] Per-class AP for active local YOLOv8s fallback.
- [ ] Per-class precision and recall.
- [x] Confusion matrix for local YOLOv8s validation/test artifacts.
- [x] Precision-recall curves for local YOLOv8s validation/test artifacts.
- [x] F1-confidence curve for local YOLOv8s validation/test artifacts.

### RD-052 — Report IoU correctly

Because object-detection mAP is already computed across IoU thresholds, do not present an unexplained standalone “IoU” value.

- [x] Report `mAP@0.50` and `mAP@0.50:0.95` for the active local YOLOv8s fallback.
- [x] Compute mean and median IoU for matched true-positive boxes at the frozen deployment threshold.
- [x] State the matching IoU threshold used to define a true positive.
- [ ] Report matched-box IoU by class.
- [ ] Visualize the distribution of matched-box IoU.
- [x] Explain that this diagnostic does not replace mAP in report caveats.

### RD-053 — Perform qualitative error analysis

- [ ] Save representative:
  - true positives;
  - false positives;
  - false negatives;
  - localization errors;
  - class-confusion errors;
  - duplicate detections.
- [ ] Analyze difficult conditions:
  - low light;
  - strong shadows;
  - wet roads;
  - faded damage;
  - tiny distant damage;
  - partial occlusion;
  - unusual camera angles;
  - repaired surfaces;
  - road markings and texture confusion.
- [ ] Identify at least the top five recurring failure modes.
- [ ] Quantify failure modes where possible.
- [ ] Add recommended data or modeling improvements for each major failure mode.

### RD-054 — Test robustness

- [ ] Evaluate on controlled transformations not used for model selection:
  - mild blur;
  - compression;
  - brightness changes;
  - resolution reduction.
- [ ] Compare relative metric degradation.
- [ ] Verify that the model does not produce excessive detections on clean-road negative images.
- [ ] Collect a small external-domain sample when licensing and time permit.
- [ ] Clearly label external-domain results as exploratory rather than part of the official test score.

---

## 9. Phase 6 — Speed, Size, and Deployment Benchmarking

### RD-060 — Create a reproducible benchmark harness

- [x] Record for the active local YOLOv8s benchmark:
  - CPU;
  - GPU;
  - RAM;
  - VRAM;
  - operating system;
  - Python version;
  - PyTorch version;
  - Ultralytics version;
  - runtime backend;
  - precision;
  - batch size;
  - image size.
- [x] Benchmark with batch size `1`.
- [x] Run a warm-up phase before timing.
- [x] Use the same fixed benchmark image set for every model.
- [x] Time enough samples to report stable statistics.
- [x] Synchronize CUDA before and after timed GPU sections.
- [x] Exclude model-loading time from steady-state inference latency.
- [x] Separately report cold-start time.
- [x] Report:
  - preprocessing time;
  - model inference time;
  - postprocessing time;
  - end-to-end latency;
  - median latency;
  - p95 latency;
  - FPS.
- [x] Define FPS as `1000 / end-to-end latency in milliseconds` for batch-size-one benchmarking.
- [x] Do not mix vendor-reported speed with locally measured speed.

### RD-061 — Measure model complexity and resources

For each final checkpoint, record:

- [x] Number of parameters for active local YOLOv8s.
- [x] FLOPs or GFLOPs at the benchmark resolution for active local YOLOv8s.
- [x] Native checkpoint size in MB for active local YOLOv8s.
- [x] Exported model size in MB for active local YOLOv8s.
- [ ] Peak CPU RAM during inference.
- [x] Peak GPU VRAM during inference for active local YOLOv8s.
- [x] Cold-start/model-load time for active local YOLOv8s.
- [x] Median and p95 end-to-end latency for active local YOLOv8s.
- [x] FPS for active local YOLOv8s.

### RD-062 — Export deployment candidates

- [x] Export the active local YOLOv8s checkpoint to ONNX.
- [x] Copy active local YOLOv8s `.pt` and `.onnx` artifacts to production export paths.
- [ ] Where supported by the target hardware, evaluate:
  - PyTorch FP32;
  - PyTorch FP16;
  - ONNX Runtime FP32;
  - ONNX Runtime quantized or FP16;
  - TensorRT FP16 on NVIDIA hardware;
  - OpenVINO on Intel hardware.
- [ ] Validate output equivalence against the native model.
- [ ] Measure metric drift after export or quantization.
- [ ] Reject an export if its accuracy loss exceeds the documented tolerance.
- [x] Save export commands and metadata.

### RD-063 — Calculate deployment efficiency

- [x] Create a current comparison table containing accuracy, latency, FPS, size, memory, and export compatibility for available models.
- [ ] Identify the Pareto frontier rather than relying only on a weighted score.
- [ ] Define hard deployment constraints before selection, such as:
  - maximum model size;
  - minimum FPS;
  - maximum p95 latency;
  - minimum recall for safety-relevant classes.
- [ ] Optionally calculate a documented composite deployment score.
- [ ] Run a sensitivity check showing how the winner changes under different score weights.

---

## 10. Phase 7 — Comparative Analysis and Model Selection

### RD-070 — Build the final comparison table

Include one row per model/configuration and at least these columns:

| Model | Variant | Image Size | Precision | Recall | F1 | mAP50 | mAP50-95 | Mean Matched IoU | Median Latency | p95 Latency | FPS | Params | Size MB | Peak VRAM | Export |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

- [x] Include current baseline/hosted results available so far.
- [x] Highlight results obtained under exactly comparable settings through notes/caveats.
- [x] Add per-class tables for Roboflow hosted evaluations and local YOLOv8s artifacts.
- [ ] Add mean ± standard deviation for multi-seed runs.
- [x] Link current rows to configs, checkpoints, endpoints, or raw run artifacts.

### RD-071 — Select the production model

- [ ] Eliminate models that fail hard latency, memory, size, or recall constraints.
- [ ] Compare remaining models on the Pareto frontier.
- [x] Choose hosted YOLO11n as the current metric leader based on available validation metrics across YOLOv8, YOLO11, and YOLO26.
- [x] Choose local YOLOv8s as the temporary active deployment model while Roboflow credits block hosted inference.
- [x] Write a decision record explaining:
  - why the model was selected;
  - why alternatives were rejected;
  - accuracy/speed trade-offs;
  - class-specific risks;
  - export/runtime choice.
- [x] Do not describe a model as “best” without stating the selection criteria.

### RD-072 — Write the comparative discussion

- [ ] Explain baseline differences among YOLOv8, YOLO11, and YOLO26.
- [ ] Quantify the effect of tuning for each model.
- [ ] Discuss which classes are easiest and hardest.
- [ ] Discuss whether newer versions consistently outperform older versions.
- [ ] Explain the impact of image resolution.
- [ ] Explain accuracy versus inference-speed trade-offs.
- [ ] Discuss model size and deployment implications.
- [ ] Analyze whether improvements are practically meaningful, not only numerically higher.
- [ ] State threats to validity:
  - dataset size;
  - label quality;
  - class imbalance;
  - domain shift;
  - seed variance;
  - benchmark hardware;
  - test-set representativeness.
- [ ] Include honest limitations and next steps.

---

## 11. Phase 8 — Inference Package

### RD-080 — Build a reusable inference service

- [x] Implement a model-loading function with configurable checkpoint and device.
- [x] Implement image preprocessing in one place.
- [x] Implement prediction with configurable:
  - confidence threshold;
  - IoU/NMS threshold where applicable;
  - selected classes;
  - maximum detections.
- [x] Implement local Ultralytics inference for the active YOLOv8s deployment model.
- [x] Implement hosted Roboflow inference for the blocked-but-preferred YOLO11n metric leader.
- [x] Add direct Roboflow serverless HTTP fallback when SDK `project.version(...).model` is unavailable.
- [x] Return structured detections containing:
  - class ID;
  - class name;
  - confidence;
  - bounding-box coordinates;
  - inference timing.
- [x] Implement annotation rendering separately from prediction logic.
- [x] Support NumPy arrays, PIL images, paths, and uploaded byte streams through app-level decoding.
- [x] Add helpful exceptions for invalid images and unsupported models.
- [x] Avoid coupling Streamlit code to training code.

### RD-081 — Add inference tests

- [x] Unit-test coordinate conversion.
- [x] Unit-test class filtering.
- [x] Unit-test confidence filtering.
- [x] Unit-test empty detections.
- [x] Unit-test annotation rendering.
- [x] Unit-test hosted Roboflow direct HTTP inference fallback and API-key redaction.
- [x] Add a smoke test using one small test image and the active production checkpoint.
- [ ] Verify deterministic results within an appropriate tolerance.

---

## 12. Phase 9 — Streamlit Deployment Application

### RD-090 — Complete the frontend-design gate

This task is mandatory before coding the final UI.

- [x] Use the frontend design skill/workflow to define a purposeful visual direction for the application.
- [x] Produce a compact design brief containing:
  - target users;
  - primary user journey;
  - visual hierarchy;
  - page layout;
  - typography;
  - color tokens;
  - spacing system;
  - component states;
  - accessibility constraints;
  - mobile/responsive behavior.
- [x] Use a road-infrastructure visual identity without relying on a generic default Streamlit appearance.
- [x] Maintain strong contrast and readable typography.
- [x] Do not use excessive gradients, animations, glass effects, or decorative elements that distract from detections.
- [x] Review the design before implementation.
- [x] Store the design brief in `reports/ui_design.md`.

**Acceptance criteria**

- [x] The UI has a consistent design system.
- [x] The most important action and result are visible without unnecessary scrolling on a standard laptop screen.
- [x] Controls are grouped logically and are not scattered across the page.
- [x] Status, warning, success, and error states are visually distinct and accessible.

### RD-091 — Build the Streamlit app shell

- [x] Configure page title, icon, layout, and sidebar.
- [x] Add a clear project header and one-sentence explanation.
- [x] Display the deployed model name and runtime.
- [x] Cache the model with `st.cache_resource`.
- [x] Keep UI rendering separate from inference services.
- [x] Add a concise model limitation and safety notice.
- [x] Add graceful loading, empty, success, and error states.

### RD-092 — Implement image input

- [x] Support drag-and-drop image upload.
- [x] Support camera capture.
- [x] Accept only documented image formats.
- [x] Validate file type, file size, and decoded image dimensions.
- [x] Correct EXIF orientation.
- [x] Convert images consistently to RGB.
- [x] Show the original image before inference.
- [x] Avoid permanently storing user-uploaded images.

### RD-093 — Implement detection controls

- [x] Add a confidence-threshold slider.
- [x] Add an IoU/NMS-threshold slider only when applicable.
- [x] Add class filtering.
- [x] Add a maximum-detections control when useful.
- [x] Add a Run Detection action.
- [x] Provide sensible defaults selected from validation results.
- [x] Explain controls through concise help text.

### RD-094 — Implement results UI

- [x] Display the annotated result prominently.
- [x] Provide original/annotated comparison.
- [x] Display:
  - total detections;
  - counts by damage type;
  - highest confidence;
  - end-to-end inference time;
  - approximate FPS;
  - active threshold settings.
- [x] Show a detection table with class, confidence, and coordinates.
- [x] Use consistent class colors.
- [x] Handle the no-damage-detected state clearly.
- [x] Add an image download button for the annotated result.
- [x] Add a CSV or JSON download for detections.

### RD-095 — Add optional real-time mode

Implement this only after stable image inference is complete.

- [ ] Evaluate whether `streamlit-webrtc` or another supported component is appropriate.
- [ ] Process frames at a controlled rate.
- [ ] Avoid blocking the UI.
- [ ] Show measured real-time FPS.
- [ ] Allow users to stop the stream immediately.
- [ ] Avoid retaining video frames.
- [ ] Disable real-time mode automatically when the hosting environment cannot support it.
- [ ] Keep image upload and camera capture as the reliable fallback.

### RD-096 — Optimize application performance

- [x] Load the model once per process.
- [x] Cache only safe, reusable resources.
- [x] Avoid rerunning inference on unrelated widget changes.
- [x] Resize very large images using a documented maximum while preserving aspect ratio.
- [ ] Measure app cold start and steady-state latency.
- [x] Use the deployment runtime selected during benchmarking.
- [ ] Confirm acceptable memory use on the target host.
- [ ] Add friendly messaging when GPU acceleration is unavailable.

### RD-097 — Test the application

- [ ] Test each supported image type.
- [ ] Test very small and very large images.
- [ ] Test corrupt and renamed non-image files.
- [ ] Test no detections.
- [ ] Test many detections.
- [ ] Test every class filter.
- [ ] Test confidence and IoU controls.
- [ ] Test model-loading failure.
- [ ] Test CPU fallback.
- [x] Test desktop and narrow/mobile layouts.
- [ ] Perform keyboard-navigation and contrast checks.
- [x] Run a local production-checkpoint smoke test for the active YOLOv8s fallback.
- [ ] Run a hosted production deployment smoke test after Roboflow serverless credits are available.

### RD-098 — Deploy the Streamlit application

- [ ] Choose and document the deployment platform.
- [ ] Pin deployment dependencies.
- [x] Add Streamlit configuration under `.streamlit/config.toml`.
- [ ] Store secrets through the platform's secret manager.
- [ ] Ensure the final checkpoint is available without an untracked local path.
- [ ] Confirm model licensing is compatible with deployment.
- [ ] Add health-check or startup verification behavior.
- [ ] Verify cold-start time, memory limit, and upload limit.
- [ ] Publish the application URL in `README.md`.
- [x] Add screenshots or a short demo recording.

---

## 13. Phase 10 — Documentation and Reporting

### RD-100 — Write the README

- [x] Add project overview and motivation.
- [ ] Add class definitions.
- [ ] Add dataset source and license.
- [ ] Add repository structure.
- [x] Add environment setup.
- [x] Add dataset preparation commands.
- [x] Add baseline training commands.
- [x] Add tuning commands.
- [x] Add evaluation and benchmark commands.
- [x] Add app launch instructions.
- [ ] Add deployment link.
- [x] Add key results table.
- [x] Add limitations and responsible-use notes.
- [ ] Add citations and acknowledgments.

### RD-101 — Create the model card

- [x] Describe the production model and runtime format.
- [x] Document intended use.
- [x] Document out-of-scope use.
- [x] Document training data and class mapping.
- [x] Report current evaluation metrics.
- [x] Report speed and hardware for the active local YOLOv8s fallback.
- [x] Describe current limitations and failure modes/caveats.
- [x] Describe threshold defaults.
- [x] Document current preprocessing.
- [ ] Document licensing.
- [x] Add the model checksum.

### RD-102 — Write the final report

Recommended structure:

1. Executive summary
2. Problem definition
3. Dataset and label audit
4. Data preparation
5. Experimental methodology
6. Baseline models
7. Hyperparameter tuning
8. Evaluation protocol
9. Quantitative results
10. Speed and deployment benchmarks
11. Qualitative error analysis
12. Comparative discussion
13. Production model selection
14. Streamlit application
15. Limitations and threats to validity
16. Future work
17. Reproducibility details
18. References

- [ ] Ensure every chart has a title, labels, units, and caption.
- [ ] Separate validation results from final test results.
- [ ] Do not cherry-pick only successful examples.
- [ ] Include failure examples.
- [ ] Ensure reported values match saved raw metrics.

### RD-103 — Add reproducibility artifacts

- [x] Save current deployment YAML configuration.
- [ ] Save split manifests.
- [ ] Save dependency lock information.
- [ ] Save environment and hardware reports.
- [x] Save current predictions and raw metrics.
- [x] Save benchmark outputs.
- [x] Save model/export checksums.
- [ ] Add one command or script that reproduces each final table.

---

## 14. Experiment Registry

Maintain this table in a CSV, database, or experiment-tracking tool.

| Run ID | Model | Variant | Stage | Dataset Version | Seed | Image Size | Epochs | Best Epoch | Config Path | mAP50 | mAP50-95 | Precision | Recall | Latency | Artifact Path | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| YOLOv8s-baseline-seed42 | YOLOv8 | s | baseline / active fallback | Roboflow v1 | 42 | 640 | 89 | 69 | `runs/YOLOv8s/baseline_config.json` | 0.421023 val / 0.583059 test | 0.234207 val / 0.346824 test | 0.645656 val / 0.621108 test | 0.403637 val / 0.535117 test | 11.71 ms median / 85.42 FPS | `models/exports/production_road_damage_model.pt` | Active local fallback; ONNX export and checksums saved. |
| YOLO11s-baseline-seed42-invalid | YOLO11 | s | incomplete local baseline | Roboflow v1 | 42 | 640 | 3 | 3 | `runs/YOLO11s/baseline_config.json` | 0.093453 val / 0.166565 test | 0.034402 val / 0.060637 test | 0.139466 val / 0.220215 test | 0.159236 val / 0.245321 test | 12.15 ms median / 82.29 FPS | `runs/YOLO11s/baseline_seed42` | Invalid for fair comparison; only 3 epochs completed. |
| roadddd-9ducw-2-yolo11n-t1 | YOLO11 | n | Roboflow fine-tune | roadddd-9ducw v2 | n/a | 640 | n/a | n/a | Roboflow hosted | 0.420338 val / 0.594240 test | 0.244293 val / 0.368296 test | n/a val / 0.637 test | n/a val / 0.608 test | hosted | `https://serverless.roboflow.com/roadddd-9ducw/2` | Current best available by validation mAP50-95; hosted inference currently blocked by Roboflow credit cap. |
| roadddd-9ducw-2-yolo26s-t2 | YOLO26 | s | Roboflow fine-tune | roadddd-9ducw v2 | n/a | 640 | n/a | n/a | Roboflow hosted | 0.389909 val / 0.523118 test | 0.228387 val / 0.317128 test | n/a val / 0.593 test | n/a val / 0.552 test | hosted | `https://serverless.roboflow.com/roadddd-9ducw/2` | Finished hosted training; underperforms YOLO11n on validation and test mAP50-95. |
| TBD | YOLO11 | s | baseline | TBD | 42 | 640 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| TBD | YOLO26 | s | baseline | TBD | 42 | 640 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |

---


## 15. Final Deliverables

- [ ] Cleaned and documented dataset configuration.
- [ ] Dataset validation and EDA reports.
- [ ] Baseline YOLOv8, YOLO11, and YOLO26 experiments.
- [ ] Hyperparameter-tuning studies for all selected families.
- [ ] Final checkpoint for each compared model.
- [ ] Test-set evaluation report.
- [ ] Speed, FPS, model-size, memory, and export benchmark report.
- [ ] Comparative analysis with model-selection rationale.
- [ ] Production model in native and selected deployment formats.
- [ ] Reusable inference package.
- [ ] Polished Streamlit application.
- [ ] Deployed app URL.
- [ ] Automated tests and CI workflow.
- [ ] README.
- [ ] Model card.
- [ ] Final technical report.
- [ ] Demo screenshots or video.
- [ ] Versioned project release.

---
