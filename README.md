# Road Damage Detection

Local training and evaluation workflow for the Roboflow road-damage dataset using the same notebook protocol that was prepared for Colab. The current production runtime is the locally trained YOLOv8s checkpoint; Roboflow-hosted YOLO11n and YOLO26s results are imported as comparison evidence.

The original Colab notebooks are still included for reference. The local workflow lives in `configs/`, `scripts/`, and `src/road_damage/`.

## Local Setup

Use Python 3.10 or 3.11. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

On a Windows NVIDIA GPU machine, install CUDA-enabled PyTorch wheels after the
editable install:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Check CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

CPU execution is supported for audits and smoke tests. Training on CPU is allowed but will be very slow; pass `--require-gpu` to training commands when you want the run to fail instead of falling back.

## Code Quality

Run the local checks before sharing changes:

```powershell
python -m ruff check app src scripts tests
python -m ruff format --check app src scripts tests
python -m pytest
```

Optional pre-commit hooks are configured for linting, formatting checks, and the
lightweight synthetic inference smoke:

```powershell
pre-commit install
pre-commit run --all-files
```

## Secrets

The Roboflow key is loaded from `ROBOFLOW_API_KEY` or from a local `.env` file. The repository ignores `.env` files and never needs the key inside notebooks, scripts, or YAML configs.

```powershell
Copy-Item .env.example .env
# Then set ROBOFLOW_API_KEY in .env
```

## Dataset

Download, convert polygon labels to detection boxes where needed, create a local Ultralytics YAML, and run the training gate:

```powershell
python scripts/download_data.py --force
python scripts/validate_data.py
```

The dataset is expected under:

```text
data/processed/road-damage-detection-bbox-v1
```

The local v1 export metadata reports a `CC BY 4.0` dataset license. Keep source
attribution with any redistributed dataset-derived artifacts, and re-check the
current Roboflow page before public release.

Current working class definitions:

| Class | Working definition |
|---|---|
| `alligator` | Interconnected fatigue cracking with a blocky or scaled pattern. |
| `block` | Large block-like damaged or patched road surface region. |
| `crack` | General visible cracking that is not clearly longitudinal, transverse, or alligator. |
| `edge` | Pavement-edge damage or deterioration along the road boundary. |
| `longitudinal` | Crack running mostly parallel to the direction of travel. |
| `pothole` | Depressed or missing pavement area forming a hole or cavity. |
| `transverse` | Crack running mostly across the lane, perpendicular to travel. |

Generated audit reports are written to `data/reports/data_audit_bbox_v1`.
EDA plots and tables are written to `reports/figures/data`.

## Repository Structure

```text
app/                  Streamlit application
configs/              Data, training, tuning, and deployment YAML files
data/                 Local raw, processed, interim, and audit data
models/               Local checkpoints and deployment exports
reports/              Evaluation outputs, EDA artifacts, final report, UI brief
runs/                 Ultralytics training runs
scripts/              Reproducible CLI entry points
src/road_damage/      Reusable package code
tests/                Unit and smoke tests
```

## Baseline Training

Run the selected local production model:

```powershell
python scripts/train.py --model YOLOv8s
```

The shared baseline config is `configs/training/baseline.yaml`. Runs are written to `runs/<model>/`. The current project scope does not include retraining local YOLO11 or YOLO26 runs; those families are represented by Roboflow-hosted trained models.

## Tuning And Final Training

Tuning is explicit so it does not start accidentally:

```powershell
python scripts/tune.py --model YOLOv8s --iterations 20 --epochs 40
```

After a YOLOv8s tuning study exists, retrain the final local model:

```powershell
python scripts/train.py --model YOLOv8s --stage final
```

## Evaluation, Benchmarking, And Export

Use the untouched test set only after the validation-based selection criteria are fixed:

```powershell
python scripts/evaluate.py
python scripts/benchmark.py
python scripts/export_model.py --export-onnx
python scripts/check_export_equivalence.py
python scripts/build_split_manifests.py
python scripts/build_final_comparison.py
python scripts/build_error_analysis.py
```

Outputs are written to `reports/final_comparison` and `models/exports`.
Split manifests are written to `data/reports/split_manifests`.
Qualitative error-analysis tables are written to `reports/final_comparison`.

## Streamlit App

Launch the app locally:

```powershell
streamlit run app/app.py
```

The app uses `configs/deployment/app.yaml` and selects the local YOLOv8s
production checkpoint at `models/exports/production_road_damage_model.pt`.
Roboflow-hosted YOLO11n remains a useful metric reference, but local YOLOv8s is
the production target for this project.

## Deployment

Chosen deployment target: Streamlit using the local YOLOv8s checkpoint. The
local deployment URL is:

```text
http://localhost:8501
```

Run the deployment smoke test:

```powershell
python scripts/deployment_smoke.py
```

Start the deployed app:

```powershell
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Docker packaging is included for hosts with Docker installed:

```powershell
docker compose up --build
```

Deployment notes are in `DEPLOYMENT.md`. Local smoke outputs and screenshots are
saved under `reports/deployment/`.

For Streamlit Community Cloud, deploy from the Cloud workspace and enter the
repository manually if the local toolbar cannot detect it:

```text
Repository: tarekamr737/Road-Damage-Object-Detection
Branch: main
Main file path: app/app.py
Python version: 3.11
```

The production `.pt` checkpoint is the only model artifact that should be
published with the app. No Roboflow secret is required for the active local
YOLOv8s runtime.

## Current Results Snapshot

The hosted YOLO11n row has the highest current validation mAP50-95, but the
selected production model is local YOLOv8s because it is available offline,
benchmarked locally, and close to the hosted metric leader:

| Model | Source | Split | mAP50 | mAP50-95 | Notes |
|---|---|---|---:|---:|---|
| YOLO11n | Roboflow hosted | valid | 0.420338 | 0.244293 | Hosted metric leader |
| YOLOv8s | Local Ultralytics | valid | 0.421022 | 0.234207 | Selected production runtime |
| YOLOv8s | Local Ultralytics | test | 0.583059 | 0.346824 | Local production test score |
| YOLO26s | Roboflow hosted | valid | 0.389909 | 0.228387 | Finished hosted training |

Active YOLOv8s benchmark on 150 validation images: median latency `11.71 ms`,
p95 latency `14.74 ms`, approximately `85.42 FPS`, and peak GPU VRAM
`78.51 MB`. Production exports are saved as:

- `models/exports/production_road_damage_model.pt`
- `models/exports/production_road_damage_model.onnx`

The Streamlit app uses the PyTorch `.pt` checkpoint. The ONNX export exists for
experimentation, but the current ONNX Runtime equivalence smoke did not pass the
documented tolerance, so ONNX is not the active deployment runtime.

See `reports/final_comparison/model_comparison_current.csv` and
`reports/final_comparison/deployment_pareto_frontier.csv` for the current
comparison and deployment Pareto summary. The full production decision record is
saved in `reports/final_comparison/production_model_decision.json`.

## Current Local Migration Status

- Local project structure, configs, scripts, and package modules are in place.
- Dependencies are installed in `.venv`; CUDA-enabled PyTorch is verified on the local NVIDIA GPU.
- Dataset download and validation have run for the local Roboflow v1 processed dataset.
- Frozen split manifests and a dependency lock file are saved locally for reproducibility.
- YOLOv8s trained, evaluated on the local test split, benchmarked, and exported locally; YOLO11n and YOLO26s hosted Roboflow metrics have been imported.
- The Streamlit app is implemented and runs local YOLOv8s inference as the production runtime.
- Local incomplete YOLO11 artifacts were removed; no local YOLO11/YOLO26 retraining is planned under the current scope.
