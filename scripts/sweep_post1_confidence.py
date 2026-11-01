"""measure how confidence filtering affects recall and false positives."""

import argparse
import csv
import json
from pathlib import Path

from perception_under_uncertainty.evaluation import IOU_THRESHOLD, match_predictions

CACHE_PATH = Path("results/post1/predictions.json")
OUTPUT_PATH = Path("runs/post1_confidence_sweep.csv")
THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.70, 0.80)


def confidence_sweep(data, thresholds=THRESHOLDS):
    annotations = data["annotations"]
    if not annotations or not data["predictions"]:
        raise ValueError("The cache needs images and prediction methods")
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Confidence thresholds must be between zero and one")
    if min(thresholds) < data["confidence_threshold"]:
        raise ValueError("Cannot recover predictions below the saved confidence cutoff")
    if not any(annotations.values()):
        raise ValueError("Recall needs at least one annotated object")

    rows = []
    for method, image_predictions in data["predictions"].items():
        for confidence in thresholds:
            tp = fp = fn = 0
            for image_name, ground_truth in annotations.items():
                # filter saved detections without repeating inference.
                predictions = [p for p in image_predictions[image_name] if p["score"] >= confidence]
                # keep box matching fixed to isolate confidence filtering.
                result = match_predictions(predictions, ground_truth, IOU_THRESHOLD)
                tp += result["tp"]
                fp += result["fp"]
                fn += result["fn"]

            rows.append(
                {
                    "method": method,
                    "confidence": confidence,
                    "iou_threshold": IOU_THRESHOLD,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "recall": tp / (tp + fn),
                    "fp_per_image": fp / len(annotations),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=CACHE_PATH, help="saved predictions to filter"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="destination csv")
    args = parser.parse_args()
    data = json.loads(args.cache.read_text(encoding="utf-8"))
    rows = confidence_sweep(data)

    method = None
    for row in rows:
        if row["method"] != method:
            method = row["method"]
            print(f"\n{method}")
        print(
            f"Conf >= {row['confidence']:.2f} | Recall {row['recall']:.1%} | "
            f"FP/image {row['fp_per_image']:.2f} | TP {row['tp']} | FP {row['fp']}"
        )

    # save measurements so plots can use the actual results.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
