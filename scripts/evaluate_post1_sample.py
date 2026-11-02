"""
Evaluate the 300 DUT sample with all four methods and comparable CPU timings
"""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sahi import AutoDetectionModel
from ultralytics import YOLO

from perception_under_uncertainty import evaluation
from perception_under_uncertainty.evaluation import IOU_THRESHOLD, match_predictions
from perception_under_uncertainty.runner import _source_details
from scripts import evaluate_post1
from scripts.evaluate_post1 import (
    CONF_THRESHOLD,
    MODEL_FILENAME,
    MODEL_REPO,
    SAHI_SETTINGS,
    predict_sahi,
    predict_yolo,
)
from scripts.select_post1_evaluation import file_sha256, inspect_pair
from scripts.sweep_post1_confidence import confidence_sweep

METHODS = ("YOLO 640", "YOLO 1280", "SAHI 640", "SAHI 640 + full image")
TIMING_BOUNDARY = (
    "loaded RGB PIL image to serialisable final predictions, including preprocessing, slicing, "
    "model calls and merging; excluding decoding, model loading, matching and file writes"
)


def validate_sample(manifest, images_dir, xml_dir):
    """Check the frozen list and every input before loading a detector."""
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported sample manifest version")
    records = manifest["images"]
    names = [record["image"] for record in records]
    if not names or len(names) != len(set(names)) or len(names) != manifest["sample_size"]:
        raise ValueError("Sample size or unique filenames do not match the manifest")
    sample_list = Path(manifest["sample_list"])
    if file_sha256(sample_list) != manifest["sample_list_sha256"]:
        raise ValueError("Frozen sample list checksum mismatch")
    if sample_list.read_text(encoding="utf-8").splitlines() != names:
        raise ValueError("Frozen list and manifest name different images")
    if set(names).intersection(manifest["audit"]["excluded"]):
        raise ValueError("The sample contains excluded images")
    for record in records:
        if (
            Path(record["image"]).name != record["image"]
            or Path(record["xml"]).name != record["xml"]
        ):
            raise ValueError("Manifest entries must be filenames")
        actual = inspect_pair(images_dir / record["image"], xml_dir / record["xml"])
        if actual != record:
            raise ValueError(f"Input differs from the frozen manifest: {record['image']}")
    return records


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def cpu_description():
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor()


def collect_predictions(records, images_dir, predictors, warmup_paths, passes=3, seed=20261102):
    """Warm up outside the clock, then time each method on the same image orders."""
    if passes < 1 or not warmup_paths:
        raise ValueError("At least one timed pass and one warm-up image are required")
    names = [record["image"] for record in records]
    if set(names).intersection(path.name for path in warmup_paths):
        raise ValueError("Warm-up images must be outside the evaluation sample")
    predictions = {method: {} for method in predictors}
    timings = []
    orders = []
    methods = list(predictors)
    for method, predict in predictors.items():
        for path in warmup_paths:
            predict(load_rgb(path))
        print(f"Warmed up {method}", flush=True)

    for pass_index in range(passes):
        order = methods[pass_index % len(methods) :] + methods[: pass_index % len(methods)]
        image_order = names.copy()
        random.Random(seed + pass_index).shuffle(image_order)
        orders.append({"pass": pass_index + 1, "methods": order, "images": image_order})
        for method in order:
            for name in image_order:
                image = load_rgb(images_dir / name)
                started = perf_counter()
                result = predictors[method](image)
                elapsed = perf_counter() - started
                timings.append(
                    {"method": method, "pass": pass_index + 1, "image": name, "seconds": elapsed}
                )
                if pass_index == 0:
                    predictions[method][name] = result
            print(f"Finished pass {pass_index + 1}/{passes} for {method}", flush=True)
    return predictions, timings, orders


def summarise(cache, timings):
    rows = []
    for method, image_predictions in cache["predictions"].items():
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for name, annotations in cache["annotations"].items():
            matched = match_predictions(image_predictions[name], annotations)
            for key in totals:
                totals[key] += matched[key]
        by_image = defaultdict(list)
        for row in timings:
            if row["method"] == method:
                by_image[row["image"]].append(row["seconds"] * 1000)
        if set(by_image) != set(cache["annotations"]):
            raise ValueError(f"Incomplete timing coverage for {method}")
        # Each image contributes once, using its median across timed passes.
        per_image_ms = [median(values) for values in by_image.values()]
        tp, fp, fn = totals.values()
        rows.append(
            {
                "method": method,
                "confidence": CONF_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
                **totals,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "fp_per_image": fp / len(cache["annotations"]),
                "median_ms": median(per_image_ms),
                "p90_ms": float(np.percentile(per_image_ms, 90, method="linear")),
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/dut_independent_eval.json"))
    parser.add_argument("--images-dir", type=Path, help="override only the local image root")
    parser.add_argument("--xml-dir", type=Path, help="override only the local annotation root")
    parser.add_argument("--output-dir", type=Path, help="new run directory, which must not exist")
    parser.add_argument(
        "--validate-only", action="store_true", help="check inputs without inference"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="test on two original images, not the new sample"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images_dir = args.images_dir or Path(manifest["images_dir"])
    xml_dir = args.xml_dir or Path(manifest["xml_dir"])
    records = validate_sample(manifest, images_dir, xml_dir)
    uavs = sum(record["annotation_count"] for record in records)
    print(f"Validated {len(records)} frozen image/XML pairs containing {uavs} UAVs", flush=True)
    if args.validate_only:
        return
    started_at = datetime.now(UTC)
    kind = "smoke" if args.smoke_test else "evaluation"
    output = args.output_dir or Path(f"runs/dut_300_{kind}_{started_at:%Y%m%dT%H%M%S%fZ}")
    if output.exists():
        parser.error("output directory already exists; choose a new directory")

    explored = [
        name for name, reason in manifest["audit"]["excluded"].items() if reason == "explored"
    ]
    if len(explored) < 7:
        parser.error("need five exploratory warm-up images and two separate smoke-test images")
    warmup_paths = [images_dir / name for name in sorted(explored)[:5]]
    if args.smoke_test:
        records = [
            inspect_pair(images_dir / name, xml_dir / f"{Path(name).stem}.xml")
            for name in sorted(explored)[5:7]
        ]
    passes = 1 if args.smoke_test else 3
    weights = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_files_only=True)
    yolo = YOLO(weights)
    sahi = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=CONF_THRESHOLD,
        image_size=640,
        device="cpu",
    )
    full_settings = {**SAHI_SETTINGS, "perform_standard_pred": True}
    predictors = {
        "YOLO 640": lambda image: predict_yolo(yolo, image, 640),
        "YOLO 1280": lambda image: predict_yolo(yolo, image, 1280),
        "SAHI 640": lambda image: predict_sahi(sahi, image),
        "SAHI 640 + full image": lambda image: predict_sahi(sahi, image, settings=full_settings),
    }
    output.mkdir(parents=True, exist_ok=False)
    cache = {
        "status": "running",
        "run_kind": kind,
        "model": MODEL_REPO,
        "confidence_threshold": CONF_THRESHOLD,
        "annotations": {record["image"]: record["annotations"] for record in records},
        "image_sizes": {record["image"]: record["image_size"] for record in records},
        "inference_settings": {
            "device": "cpu",
            "batch_size": 1,
            "yolo_image_sizes": [640, 1280],
            "sahi_image_size": 640,
            "sahi": SAHI_SETTINGS,
            "sahi_full_image": full_settings,
        },
        "timing_settings": {
            "boundary": TIMING_BOUNDARY,
            "clock": "time.perf_counter",
            "passes": passes,
            "warmup_calls_per_method": len(warmup_paths),
            "warmup_image_sha256": {path.name: file_sha256(path) for path in warmup_paths},
            "prediction_pass": 1,
            "summary": "median and linear p90 across per-image median times over all passes",
            "method_order": "rotate one position per pass",
            "image_order": "shuffle with sample seed + zero-based pass index, shared by methods",
        },
        "provenance": {
            "started_at_utc": started_at.isoformat(),
            "sample_manifest_sha256": file_sha256(args.manifest),
            "model_filename": MODEL_FILENAME,
            "weights_sha256": file_sha256(Path(weights)),
            "model_snapshot": Path(weights).parent.name,
            "source": _source_details(Path.cwd()),
            "source_sha256": {
                path.name: file_sha256(path)
                for path in (
                    Path(__file__),
                    Path(evaluate_post1.__file__),
                    Path(evaluation.__file__),
                )
            },
            "lockfile_sha256": file_sha256(Path("uv.lock")),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": cpu_description(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "packages": {
                name: version(name)
                for name in ("ultralytics", "sahi", "torch", "torchvision", "numpy", "pillow")
            },
        },
    }
    # An interrupted run remains explicitly incomplete rather than looking like a valid cache.
    receipt = output / "run_manifest.json"
    receipt.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    predictions, timings, orders = collect_predictions(
        records, images_dir, predictors, warmup_paths, passes=passes, seed=manifest["seed"]
    )
    cache["predictions"] = predictions
    cache["timing_orders"] = orders
    # Ultralytics can set thread counts lazily during predictor setup in warm-up.
    cache["provenance"]["torch_threads"] = torch.get_num_threads()
    cache["provenance"]["torch_interop_threads"] = torch.get_num_interop_threads()
    cache["status"] = "complete"
    summary = summarise(cache, timings)
    write_csv(output / "latencies.csv", timings)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "confidence_sweep.csv", confidence_sweep(cache))
    (output / "predictions.json").write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    receipt.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(
            f"{row['method']}: TP={row['tp']}, FP={row['fp']}, FN={row['fn']}, "
            f"recall={row['recall']:.2%}, precision={row['precision']:.2%}, "
            f"median={row['median_ms']:.1f}ms, p90={row['p90_ms']:.1f}ms"
        )
    print(f"Saved {kind} results to {output}")


if __name__ == "__main__":
    main()
