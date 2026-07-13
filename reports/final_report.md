# Road Damage Detection Technical Report

## Technical Summary

The selected production runtime is the local YOLOv8s PyTorch checkpoint at
`models/exports/production_road_damage_model.pt`. Hosted Roboflow YOLO11n is the
current metric leader by validation mAP50-95 (`0.244293` versus YOLOv8s
`0.234207`), but it is not the production model because hosted inference is
credit-blocked and downloadable `.pt` weights are unavailable with the current
permissions.

The production YOLOv8s model was evaluated on the untouched local test split and
reached precision `0.621108`, recall `0.535117`, F1 `0.574915`, mAP50
`0.583059`, and mAP50-95 `0.346824`. It is fast enough for local image
inspection on the measured GPU path: median end-to-end latency was `11.71 ms`,
p95 latency was `14.74 ms`, and the batch-size-one benchmark implies about
`85.42 FPS`.

The main risk is not runtime speed; it is missed detections and class imbalance.
At confidence `0.25` and IoU `0.50`, the YOLOv8s test prediction export produced
`716` false positives and `1,047` false negatives across `681` test images.
The `edge` class has only `22` total objects in the local dataset and no test
objects in the preserved split, so edge performance should not be treated as
reliable.

## Validation Comparison Shows a Small Metric Gap

Validation mAP50-95 is the primary selection metric for comparing available
trained outputs. The comparison is operational rather than a pure architecture
study because YOLOv8s is local Ultralytics training on the processed Roboflow v1
dataset, while YOLO11n and YOLO26s are hosted Roboflow fine-tunes on project
version 2.

| Model | Variant | Source | Validation mAP50 | Validation mAP50-95 | Selection Note |
|---|---|---|---:|---:|---|
| YOLO11n | n | Roboflow hosted | 0.420338 | 0.244293 | Metric leader, not deployable locally now |
| YOLOv8s | s | Local Ultralytics | 0.421022 | 0.234207 | Selected production runtime |
| YOLO26s | s | Roboflow hosted | 0.389909 | 0.228387 | Below YOLO11n and YOLOv8s on validation mAP50-95 |

YOLO11n leads YOLOv8s by `0.010086` mAP50-95 on validation. That gap is small
enough that deployment availability, reproducibility, and local latency dominate
the production decision for the current sprint. YOLO26s does not improve the
available metric frontier.

## Test Results Confirm YOLOv8s Is Usable but Recall-Limited

Test metrics are reported separately from validation metrics. The local YOLOv8s
test split was not used as the primary model-selection signal; it is used here
to characterize the selected production model.

| Metric | YOLOv8s Test Value |
|---|---:|
| Precision | 0.621108 |
| Recall | 0.535117 |
| F1 | 0.574915 |
| mAP50 | 0.583059 |
| mAP50-95 | 0.346824 |
| Mean matched IoU | 0.751943 |
| Median matched IoU | 0.774968 |
| Matched true positives | 957 |
| Parameters | 11,128,293 |
| Checkpoint size | 21.47 MB |

The localization quality for matched detections is reasonable: median matched
IoU is `0.774968`, and the 10th percentile matched IoU is `0.571548`. The
recall value still matters operationally: a road-inspection assistant should be
treated as triage support, not as a complete defect inventory.

## Class Performance Is Uneven

Per-class YOLOv8s test mAP50-95 shows a wide spread. Larger or visually distinct
classes such as `block` and `crack` are stronger, while thin or visually subtle
damage classes such as `longitudinal` remain weak.

| Class | YOLOv8s Test mAP50-95 | Dataset Objects |
|---|---:|---:|
| block | 0.647708 | 438 |
| crack | 0.534507 | 2,115 |
| pothole | 0.355116 | 8,583 |
| edge | 0.346824 | 22 |
| transverse | 0.316075 | 677 |
| alligator | 0.148619 | 5,915 |
| longitudinal | 0.078919 | 2,867 |

The `edge` score is not decision-grade because the preserved test split contains
no edge objects. The very low `longitudinal` score is more actionable: that
class has enough objects to indicate a true model weakness, likely because many
longitudinal cracks are narrow, low contrast, and easy to confuse with road
texture.

## Deployment Efficiency Favors the Local Checkpoint

The measured deployment candidate is the local YOLOv8s PyTorch checkpoint. The
benchmark used 150 validation images, batch size 1, image size 640, and the
local NVIDIA RTX 3050 Ti Laptop GPU.

| Runtime Metric | YOLOv8s Local |
|---|---:|
| Median latency | 11.71 ms |
| p95 latency | 14.74 ms |
| Mean preprocessing | 1.81 ms |
| Mean inference | 10.41 ms |
| Mean postprocessing | 1.12 ms |
| FPS from median latency | 85.42 |
| Peak benchmark VRAM | 78.51 MB |
| Model load time | 43.45 ms |
| Cold first inference | 1,873.87 ms |

The deployment Pareto summary marks YOLOv8s as the only candidate with complete
local latency, FPS, and size measurements. Hosted YOLO11n and YOLO26s are
incomplete for deployment Pareto comparison because their local checkpoint size
and local runtime measurements are unavailable. YOLOv8s passes the configured
size, FPS, and p95 latency constraints, but its validation recall is below the
current `0.50` selection constraint. The decision to use YOLOv8s is therefore an
explicit operational choice: use the locally available model now, while tracking
recall as the next improvement target.

## Qualitative Error Analysis Includes Failures

The qualitative review is generated from saved YOLOv8s test predictions, not
hand-picked screenshots. At confidence `0.25` and IoU `0.50`, the generated
error table covers all `681` test images:

| Error Analysis Metric | Count |
|---|---:|
| Images with any false positive | 336 |
| Images with any false negative | 393 |
| Images with at least one matched detection | 530 |
| Total false positives | 716 |
| Total false negatives | 1,047 |

Representative failure examples are saved in
`reports/final_comparison/error_analysis_examples.csv`. The largest false
positive example is
`pa3611_jpg.rf.2f36ae32660850fbf152902ed74e958b.jpg`, with `14` false positives
and `8` false negatives. The largest false negative example is
`pa3111_jpg.rf.5b3684b2dc8a0b32cc77a3ee424108e6.jpg`, with `14` missed ground
truth objects. Low-localization examples begin near the IoU threshold, such as
`al687_jpg.rf.9ccb0b6c59fb637b05db2a1d5ab9e7a6.jpg` with matched IoU
`0.500049`.

The same table also includes strong-match examples, such as
`img-23_jpg.rf.3f89f88c175830959dbcd5ecfcb84487.jpg`, where five pothole ground
truth objects were matched with no false positives or false negatives. Keeping
both failure and strong-match rows prevents the qualitative review from
cherry-picking only successful detections.

## Scope, Data, and Metrics

The local dataset is the processed Roboflow v1 YOLO detection export. The data
audit recorded `4,930` train images, `1,146` validation images, and `681` test
images, with `15,025`, `3,588`, and `2,004` objects respectively. The validation
gate found zero critical validation errors and zero exact cross-split duplicate
groups.

The retained classes are `alligator`, `block`, `crack`, `edge`,
`longitudinal`, `pothole`, and `transverse`. Class imbalance is material:
`pothole` has `8,583` objects, while `edge` has only `22`. Box-scale imbalance
also matters: `51.08%` of pothole boxes are under 1% normalized image area, and
`37.60%` of longitudinal boxes are under that same threshold.

Metrics use the saved evaluation artifacts:

- Precision, recall, F1, mAP50, and mAP50-95 come from Ultralytics or Roboflow
  model-evaluation exports.
- Matched IoU uses confidence `0.25`, same-class greedy matching, and IoU
  threshold `0.50`.
- Latency and FPS come from batch-size-one local benchmarking and exclude model
  loading from steady-state latency.
- ONNX equivalence is a smoke check, not a replacement for full exported-model
  validation.

## Methodology and Experimental Controls

The current local baseline uses pretrained YOLOv8s weights, image size `640`,
seed `42`, the preserved local train/validation/test split, and local
Ultralytics training/evaluation scripts. The hosted YOLO11n and YOLO26s rows are
imported from Roboflow model evaluation because local YOLO11/YOLO26 retraining
is out of scope for the current user decision.

The comparison should not be read as proof that one YOLO family is universally
better. It answers a narrower production question: given one local YOLOv8s
checkpoint and two hosted Roboflow trained outputs, which model can be used now
in the Streamlit application with the best balance of accuracy, availability,
and deployment constraints?

## Export and Runtime Choice

Both PyTorch `.pt` and ONNX exports exist in `models/exports`, but the app uses
the PyTorch checkpoint. ONNX Runtime equivalence checked 20 images and did not
pass the configured tolerance: PyTorch produced `62` detections, ONNX produced
`64`, `12` images failed the tolerance, the maximum confidence delta was
`0.168708`, and the maximum box absolute delta was `1,331.70 px`.

The active runtime remains PyTorch because it is the evaluated path used by the
Streamlit app and because the ONNX drift is too large for a responsible runtime
switch.

## Limitations, Uncertainty, and Robustness Checks

- The local/hosted model comparison mixes dataset versions, training
  environments, model variants, and artifact availability.
- The hosted YOLO11n metric leader cannot currently be used as the app runtime
  because Roboflow serverless inference returns HTTP 402 `credit_cap_exceeded`.
- Hosted Roboflow weights are not downloadable with the current API
  permissions, so local latency, size, and export checks are unavailable for
  YOLO11n and YOLO26s.
- The local preserved split has no `edge` objects in the test split, making edge
  conclusions unreliable.
- Hyperparameter tuning and final YOLOv8 retraining are still optional/open, so
  the current YOLOv8s model should be treated as a strong baseline rather than a
  fully optimized final model.
- CPU RAM and deployed-host memory have not yet been measured; the benchmark
  currently records peak GPU VRAM.
- Robustness under blur, compression, brightness shifts, resolution reduction,
  and external-domain road imagery remains open.

## Recommended Next Steps

1. Prioritize recall improvement for YOLOv8s, especially `longitudinal`,
   `alligator`, and dense pothole scenes.
2. Perform a focused label-quality audit for confusing classes such as
   `alligator` versus `crack`, `edge` versus road boundaries, and patched
   surfaces versus `block`.
3. Add a robustness evaluation suite for blur, compression, brightness, and
   lower-resolution images before public deployment.
4. Keep PyTorch `.pt` as the production runtime until ONNX equivalence passes
   on a broader validation sample.
5. If Roboflow credits or export permissions change, re-evaluate YOLO11n with
   local latency, size, and qualitative error-analysis artifacts before
   switching the production model.

## Further Questions

- Would a validation-tuned confidence threshold improve recall enough without
  creating too many false positives for the app workflow?
- Are the most severe false negatives caused by label ambiguity, small objects,
  low contrast, or model capacity?
- Would a higher image size, such as 960, improve thin-crack recall within the
  target latency budget?
- Should the preserved split be revised or supplemented so `edge` appears in
  validation and test evaluation?

## Audited Artifacts

- `reports/final_comparison/model_comparison_current.csv`
- `reports/final_comparison/deployment_pareto_frontier.csv`
- `reports/final_comparison/per_class_map50_95.csv`
- `reports/final_comparison/matched_iou_summary.csv`
- `reports/final_comparison/benchmark.csv`
- `reports/final_comparison/error_analysis_summary.json`
- `reports/final_comparison/error_analysis_examples.csv`
- `reports/final_comparison/onnx_equivalence_summary.json`
- `data/reports/data_audit_bbox_v1/summary.json`
