from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from road_damage.data.dataset import build_ultralytics_yaml
from road_damage.data.validation import validate_dataset
from road_damage.training.runtime import load_config
from road_damage.utils.paths import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local dataset EDA artifacts.")
    parser.add_argument("--config", default="configs/data/roboflow.yaml")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--figures-dir", default="reports/figures/data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(args.config)
    dataset_dir = resolve_path(
        args.dataset_dir or config["processed_dataset_dir"],
        root,
    )
    data_yaml = (
        resolve_path(args.data_yaml, root)
        if args.data_yaml
        else build_ultralytics_yaml(dataset_dir)
    )
    figures_dir = ensure_dir(resolve_path(args.figures_dir, root))
    report = validate_dataset(dataset_dir, data_yaml)
    images_df = pd.DataFrame(report.image_records)
    boxes_df = pd.DataFrame(report.box_records)

    if boxes_df.empty:
        raise RuntimeError("No valid boxes found for EDA.")

    class_split_counts = pd.crosstab(boxes_df["class_name"], boxes_df["split"])
    class_split_counts["total"] = class_split_counts.sum(axis=1)
    class_split_counts.sort_values("total", ascending=False).to_csv(
        figures_dir / "class_distribution.csv"
    )

    ax = class_split_counts.drop(columns="total").plot(kind="bar", figsize=(13, 5))
    ax.set_title("Road-damage object count by class and split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Bounding boxes")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=180)
    plt.close()

    clipped_area = boxes_df["area"].clip(upper=boxes_df["area"].quantile(0.99))
    ax = clipped_area.plot(kind="hist", bins=50, figsize=(10, 4))
    ax.set_title("Normalized bounding-box area distribution up to 99th percentile")
    ax.set_xlabel("Normalized box area")
    plt.tight_layout()
    plt.savefig(figures_dir / "box_area_distribution.png", dpi=180)
    plt.close()

    ax = images_df["objects"].plot(kind="hist", bins=50, figsize=(10, 4))
    ax.set_title("Objects per image")
    ax.set_xlabel("Object count")
    plt.tight_layout()
    plt.savefig(figures_dir / "objects_per_image.png", dpi=180)
    plt.close()

    valid_images = images_df.dropna(subset=["width", "height"]).copy()
    valid_images["aspect_ratio"] = valid_images["width"] / valid_images["height"]
    ax = valid_images.plot.scatter(x="width", y="height", figsize=(8, 6), alpha=0.35)
    ax.set_title("Image dimensions")
    ax.set_xlabel("Width px")
    ax.set_ylabel("Height px")
    plt.tight_layout()
    plt.savefig(figures_dir / "image_dimensions.png", dpi=180)
    plt.close()

    class_statistics = (
        boxes_df.groupby("class_name")
        .agg(
            boxes=("class_name", "size"),
            median_area=("area", "median"),
            mean_area=("area", "mean"),
            median_aspect_ratio=("aspect_ratio", "median"),
        )
        .sort_values("boxes", ascending=False)
    )
    class_statistics.to_csv(figures_dir / "class_statistics.csv")

    box_dimension_statistics = (
        boxes_df.groupby("class_name")
        .agg(
            boxes=("class_name", "size"),
            median_width=("width", "median"),
            median_height=("height", "median"),
            median_area=("area", "median"),
            median_aspect_ratio=("aspect_ratio", "median"),
            p10_area=("area", lambda values: values.quantile(0.10)),
            p90_area=("area", lambda values: values.quantile(0.90)),
        )
        .sort_values("boxes", ascending=False)
    )
    box_dimension_statistics.to_csv(figures_dir / "box_dimension_statistics.csv")

    def size_bucket(area: float) -> str:
        if area < 0.01:
            return "small_area_lt_1pct"
        if area < 0.05:
            return "medium_area_1pct_to_5pct"
        return "large_area_gte_5pct"

    boxes_df["size_bucket"] = boxes_df["area"].map(size_bucket)
    size_bins = (
        pd.crosstab(boxes_df["class_name"], boxes_df["size_bucket"], normalize="index")
        .mul(100)
        .round(2)
    )
    size_bins.to_csv(figures_dir / "box_size_bins_percent.csv")

    image_class_presence = (
        boxes_df.assign(present=1)
        .pivot_table(
            index="image",
            columns="class_name",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .astype(int)
    )
    cooccurrence = image_class_presence.T.dot(image_class_presence)
    cooccurrence.to_csv(figures_dir / "class_cooccurrence.csv")

    images_df.sort_values("objects", ascending=False).head(50).to_csv(
        figures_dir / "dense_images_top50.csv",
        index=False,
    )

    valid_images.sort_values("aspect_ratio", ascending=False).head(25).to_csv(
        figures_dir / "wide_images_top25.csv",
        index=False,
    )
    valid_images.sort_values("aspect_ratio", ascending=True).head(25).to_csv(
        figures_dir / "tall_images_top25.csv",
        index=False,
    )

    print(f"Wrote EDA artifacts to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
