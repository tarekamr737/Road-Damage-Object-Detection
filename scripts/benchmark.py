from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from road_damage.data.dataset import find_validation_folder, image_paths
from road_damage.training.runtime import (
    clear_cuda_cache,
    load_config,
    peak_vram_mb,
    reset_peak_memory,
    resolve_device,
    sync_cuda,
    write_environment,
)
from road_damage.utils.paths import ensure_dir, resolve_path
from road_damage.utils.ultralytics_env import configure_ultralytics_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batch-1 inference speed.")
    parser.add_argument("--config", default="configs/training/baseline.yaml")
    parser.add_argument(
        "--deployment-config",
        default="configs/deployment/selection.yaml",
    )
    parser.add_argument("--output-dir", default="reports/final_comparison")
    parser.add_argument("--benchmark-images", type=int, default=150)
    parser.add_argument("--warmup-runs", type=int, default=10)
    return parser.parse_args()


def find_checkpoints(config: dict, root: Path) -> dict[str, Path]:
    runs_root = resolve_path(config.get("runs_root", "runs"), root)
    checkpoints: dict[str, Path] = {}
    for model_tag in config["model_specs"]:
        final = (
            runs_root
            / model_tag
            / config.get("final_name", "final_tuned_seed42")
            / "weights"
            / "best.pt"
        )
        baseline = (
            runs_root
            / model_tag
            / config.get("baseline_name", "baseline_seed42")
            / "weights"
            / "best.pt"
        )
        if final.exists():
            checkpoints[model_tag] = final
        elif baseline.exists():
            checkpoints[model_tag] = baseline
        else:
            print(f"Missing checkpoint: {model_tag}")
    if not checkpoints:
        raise FileNotFoundError("No trained checkpoints found under runs/.")
    return checkpoints


def benchmark_model(
    model: Any,
    images: list[np.ndarray],
    imgsz: int,
    confidence: float,
    device: str | int,
    warmup_runs: int,
) -> dict:
    for index in range(warmup_runs):
        model.predict(
            source=images[index % len(images)],
            imgsz=imgsz,
            conf=confidence,
            device=device,
            verbose=False,
        )

    sync_cuda()
    gc.collect()
    clear_cuda_cache()
    reset_peak_memory()

    wall: list[float] = []
    preprocessing: list[float] = []
    inference: list[float] = []
    postprocessing: list[float] = []

    for image in images:
        sync_cuda()
        started = time.perf_counter()
        result = model.predict(
            source=image,
            imgsz=imgsz,
            conf=confidence,
            device=device,
            verbose=False,
        )[0]
        sync_cuda()
        wall.append((time.perf_counter() - started) * 1000)
        preprocessing.append(float(result.speed.get("preprocess", np.nan)))
        inference.append(float(result.speed.get("inference", np.nan)))
        postprocessing.append(float(result.speed.get("postprocess", np.nan)))

    median = float(np.median(wall))
    return {
        "benchmark_images": len(images),
        "median_wall_latency_ms": median,
        "mean_wall_latency_ms": float(np.mean(wall)),
        "p95_wall_latency_ms": float(np.percentile(wall, 95)),
        "fps_from_median_latency": 1000.0 / median if median > 0 else 0.0,
        "mean_preprocess_ms": float(np.nanmean(preprocessing)),
        "mean_inference_ms": float(np.nanmean(inference)),
        "mean_postprocess_ms": float(np.nanmean(postprocessing)),
        "peak_vram_mb": peak_vram_mb(),
    }


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    config = load_config(args.config)
    deployment_config = load_config(args.deployment_config)
    data_yaml = resolve_path(config["data_yaml"], root)
    dataset_dir = data_yaml.parent
    output_dir = ensure_dir(resolve_path(args.output_dir, root))
    reports_root = ensure_dir(resolve_path(config.get("reports_root", "reports"), root))
    write_environment(reports_root / "environment.json")
    configure_ultralytics_env(root)
    from ultralytics import YOLO

    valid_name = find_validation_folder(dataset_dir)
    candidate_paths = image_paths(dataset_dir / valid_name / "images")
    benchmark_paths = candidate_paths[
        : min(args.benchmark_images, len(candidate_paths))
    ]
    if not benchmark_paths:
        raise RuntimeError("No validation images found for benchmarking.")

    benchmark_arrays = [
        np.asarray(Image.open(path).convert("RGB")) for path in benchmark_paths
    ]
    checkpoints = find_checkpoints(config, root)
    device = resolve_device(config.get("device", "auto"))
    imgsz = int(
        deployment_config.get("image_size", config["train_args"].get("imgsz", 640))
    )
    confidence = float(deployment_config.get("deploy_confidence_threshold", 0.25))

    rows: list[dict] = []
    for model_tag, checkpoint in checkpoints.items():
        print(f"Benchmarking {model_tag}...")
        gc.collect()
        clear_cuda_cache()
        load_start = time.perf_counter()
        model = YOLO(str(checkpoint))
        model_load_ms = (time.perf_counter() - load_start) * 1000

        sync_cuda()
        cold_start = time.perf_counter()
        model.predict(
            source=benchmark_arrays[0],
            imgsz=imgsz,
            conf=confidence,
            device=device,
            verbose=False,
        )
        sync_cuda()
        cold_first_inference_ms = (time.perf_counter() - cold_start) * 1000

        row = benchmark_model(
            model,
            benchmark_arrays,
            imgsz,
            confidence,
            device,
            args.warmup_runs,
        )
        row.update(
            {
                "model": model_tag,
                "checkpoint": str(checkpoint),
                "model_load_ms": model_load_ms,
                "cold_first_inference_ms": cold_first_inference_ms,
            }
        )
        rows.append(row)

    benchmark_df = pd.DataFrame(rows).sort_values("median_wall_latency_ms")
    benchmark_df.to_csv(output_dir / "benchmark.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_images": len(benchmark_arrays),
        "image_size": imgsz,
        "confidence_threshold": confidence,
        "output": str(output_dir / "benchmark.csv"),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(benchmark_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
