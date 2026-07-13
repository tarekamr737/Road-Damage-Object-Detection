# Updated Road Damage YOLO Colab Notebooks

> Local migration note: the Colab notebooks are now reference material. For local
> training, use the scripts and configs documented in `README.md`.

## Run order

1. `01_Train_YOLOv8s.ipynb`
2. `02_Train_YOLO11s.ipynb`
3. `03_Train_YOLO26s.ipynb`
4. `04_Final_Comparison_Benchmark_Export.ipynb`
5. `05_Streamlit_Deployment.ipynb`

Run the updated dataset notebook first:

`00_Data_Setup_and_Audit_UPDATED.ipynb`

## Shared processed dataset

All training and comparison notebooks use:

`/content/drive/MyDrive/RoadDamageYOLO/data/road-damage-detection-bbox-v1`

They copy it to:

`/content/road-damage-detection-bbox-v1`

and generate a local `data_colab.yaml`.

## Expected dataset counts

- Train: 4,930 images
- Validation: 1,146 images
- Test: 681 images

The preflight verifies that every required images/labels directory exists and that
the split counts are nonzero before training starts.

## Training sequence

Complete all three fair baselines first. Hyperparameter tuning is disabled by default.
After reviewing the baseline comparison, enable the same tuning budget in each model
notebook. Finally, run notebook 04 once on the frozen test set, then notebook 05.
