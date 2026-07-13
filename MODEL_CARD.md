# Model Card

This model card is provisional. It reflects the selected local production model
after importing Roboflow-hosted YOLO11 and YOLO26 results for comparison.

## Active Model

- Model: `YOLOv8s-baseline-seed42`
- Checkpoint: `models/exports/production_road_damage_model.pt`
- ONNX export: `models/exports/production_road_damage_model.onnx`
- Family: YOLOv8
- Variant: small
- Runtime: local Ultralytics/PyTorch inference
- Dataset version: local processed Roboflow v1
- Input size: 640 x 640
- Default confidence threshold: 0.25
- Checkpoint SHA256:
  `9c51b412886730363d100577098a04fc92c7e5a6c0481b786a796ee3de9d5d13`
- ONNX SHA256:
  `65f3053c0af4ba4a57d1bde81fba4db447161bfbe311f8516d71d22404390176`

Hosted YOLO11n remains the metric leader by validation mAP50-95, but local
YOLOv8s is the production runtime because it is available offline, benchmarked
locally, and close to the hosted result.

## Intended Use

Road-damage detection in still road-surface images for inspection support, education, and project review.

## Out Of Scope

The system is not a substitute for professional engineering assessment, safety inspection, or legal roadway certification.

## Training Data

Dataset source and version are defined in `configs/data/roboflow.yaml`. The processed dataset and audit reports are generated locally and are not committed.

The local v1 export metadata reports the dataset license as `CC BY 4.0`.
Attribution and redistribution requirements should be re-checked against the
current Roboflow project page before publishing model weights or dataset-derived
artifacts outside this local project.

## Evaluation

Current selection uses validation mAP50-95, with test metrics recorded only for
context. The current ranking is:

| Model | Split | mAP50 | mAP50-95 | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| YOLO11n hosted | valid | 0.420338 | 0.244293 | n/a | n/a | n/a |
| YOLOv8s local | valid | 0.421022 | 0.234207 | 0.645656 | 0.403637 | 0.496736 |
| YOLOv8s local | test | 0.583059 | 0.346824 | 0.621108 | 0.535117 | 0.574915 |
| YOLO26s hosted | valid | 0.389909 | 0.228387 | n/a | n/a | n/a |
| YOLO11n hosted | test | 0.594240 | 0.368296 | 0.637 | 0.608 | 0.614 |
| YOLO26s hosted | test | 0.523118 | 0.317128 | 0.593 | 0.552 | 0.563 |

Local YOLOv8s benchmark on 150 validation images:

- Median latency: 11.71 ms
- p95 latency: 14.74 ms
- Approximate FPS: 85.42
- Peak GPU VRAM during benchmark: 78.51 MB
- Matched true-positive IoU on local test predictions: mean 0.751943,
  median 0.774968 at confidence 0.25 and matching IoU threshold 0.50

Detailed comparison files are in `reports/final_comparison/`.

## Limitations

- This is an operational production comparison rather than a same-environment
  architecture study: YOLOv8s was trained locally on the processed v1 dataset,
  while YOLO11n and YOLO26s are hosted Roboflow fine-tunes on `roadddd-9ducw`
  v2.
- YOLOv8s is the selected production model, not the highest validation
  mAP50-95 row.
- The hosted weights are not downloadable with the current API permissions.
- ONNX export succeeded without simplification because `onnxslim` was not
  installed and package download was unavailable in the current sandbox.
- The ONNX Runtime equivalence smoke currently fails the documented tolerance
  on the validation sample (`reports/final_comparison/onnx_equivalence_summary.json`),
  so the active deployment runtime remains the PyTorch `.pt` checkpoint.
- Roboflow serverless inference currently returns HTTP 402 because the
  workspace credit cap is exceeded; this is documented for hosted comparison
  context, but the app does not depend on hosted inference.
- The `edge` class has too few/no evaluation targets for meaningful conclusions.
