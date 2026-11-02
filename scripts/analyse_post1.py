"""
Inspect missed targets, object size and errors using saved predictions
"""

import argparse
import json
from pathlib import Path

from perception_under_uncertainty.evaluation import IOU_THRESHOLD, box_iou, match_predictions

CACHE_PATH = Path("results/post1/predictions.json")


def analyse(data):
    annotations = data["annotations"]
    prediction_cache = data["predictions"]
    image_sizes = data["image_sizes"]
    matches = {}

    def size_at_640(image_name, index):
        # square-root area after whole-image resizing, before padding
        width, height = image_sizes[image_name]
        scale = min(640 / width, 640 / height)
        x1, y1, x2, y2 = annotations[image_name][index]["box"]
        return ((x2 - x1) * (y2 - y1)) ** 0.5 * scale

    print(f"Model: {data['model']}")
    print(f"Images: {len(annotations)}")
    print(f"Confidence: {data['confidence_threshold']}, IoU: {IOU_THRESHOLD}")

    for label, image_predictions in prediction_cache.items():
        matches[label] = {}
        totals = {"tp": 0, "fp": 0, "fn": 0}
        print(f"\n--- {label}: missed targets ---")

        for name, ground_truth in annotations.items():
            predictions = image_predictions[name]
            result = match_predictions(predictions, ground_truth)
            matches[label][name] = result["matched_indices"]
            for key in totals:
                totals[key] += result[key]

            for index, annotation in enumerate(ground_truth):
                if index not in result["matched_indices"]:
                    best_iou = max(
                        (box_iou(p["box"], annotation["box"]) for p in predictions),
                        default=0.0,
                    )
                    print(f"{label} {name} UAV#{index + 1}: best IoU = {best_iou:.3f}")

        tp, fp, fn = totals.values()
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        print(f"TP={tp}, FP={fp}, FN={fn}")
        print(f"Recall={recall:.2%}, Precision={precision:.2%}")
        print(f"FP/image={fp / len(annotations):.2f}")

    comparisons = [("YOLO 1280", "YOLO 640"), ("SAHI 640", "YOLO 640")]
    full_image_label = "SAHI 640 + full image"
    if full_image_label in matches:
        comparisons.append((full_image_label, "SAHI 640"))
    for label, baseline_label in comparisons:
        print(f"\n=== GAINED / LOST: {label} vs {baseline_label} ===")
        baseline = matches[baseline_label]
        total_gained = total_lost = total_fp_delta = 0
        if label == full_image_label:
            print("GT indices are zero-based; UAV# is index + 1. FP delta is full minus sliced.")
        for name in annotations:
            gained = matches[label][name] - baseline[name]
            lost = baseline[name] - matches[label][name]
            total_gained += len(gained)
            total_lost += len(lost)
            for index in sorted(gained):
                print(f"GAIN {name} UAV#{index + 1}: {size_at_640(name, index):.1f}px")
            for index in sorted(lost):
                print(f"LOST {name} UAV#{index + 1}: {size_at_640(name, index):.1f}px")

            if label == full_image_label:
                sliced_predictions = prediction_cache[baseline_label][name]
                full_predictions = prediction_cache[label][name]
                sliced_fp = len(sliced_predictions) - len(baseline[name])
                full_fp = len(full_predictions) - len(matches[label][name])
                fp_delta = full_fp - sliced_fp
                total_fp_delta += fp_delta
                boxes_changed = sorted(p["box"] for p in sliced_predictions) != sorted(
                    p["box"] for p in full_predictions
                )
                print(
                    f"{name}: gained GT={sorted(gained)}, lost GT={sorted(lost)}, "
                    f"FP={sliced_fp}->{full_fp} ({fp_delta:+d}), boxes changed={boxes_changed}"
                )
                for index, annotation in enumerate(annotations[name]):
                    size = size_at_640(name, index)
                    if size < 64 and name not in ("00605.jpg", "00967.jpg"):
                        continue
                    overlaps = [
                        max(
                            (box_iou(p["box"], annotation["box"]) for p in predictions),
                            default=0.0,
                        )
                        for predictions in (sliced_predictions, full_predictions)
                    ]
                    print(
                        f"  GT[{index}] / UAV#{index + 1}: {size:.1f}px, "
                        f"matched={index in baseline[name]}->{index in matches[label][name]}, "
                        f"best IoU={overlaps[0]:.4f}->{overlaps[1]:.4f}"
                    )

        if label == full_image_label:
            tp_delta = sum(len(indices) for indices in matches[label].values()) - sum(
                len(indices) for indices in baseline.values()
            )
            assert total_gained - total_lost == tp_delta
            print(f"TOTAL: gained={total_gained}, lost={total_lost}, FP delta={total_fp_delta:+d}")

    print("\n=== RECALL BY UAV SIZE ===")
    bins = (("<16", 0, 16), ("16-31", 16, 32), ("32-63", 32, 64), ("64+", 64, float("inf")))
    for group, lower, upper in bins:
        objects = [
            (name, index)
            for name, boxes in annotations.items()
            for index in range(len(boxes))
            if lower <= size_at_640(name, index) < upper
        ]
        if not objects:
            continue
        print(f"\n{group}px ({len(objects)} UAVs)")
        for label in matches:
            detected = sum(index in matches[label][name] for name, index in objects)
            print(f"{label}: {detected}/{len(objects)} = {detected / len(objects):.1%}")

    print(f"\n=== FALSE POSITIVE BREAKDOWN: IoU >= {IOU_THRESHOLD} ===")
    error_summaries = {}
    for label, image_predictions in prediction_cache.items():
        counts = {"duplicate": 0, "localisation_candidate": 0, "background_candidate": 0}
        total_fp = 0
        for name, ground_truth in annotations.items():
            result = match_predictions(image_predictions[name], ground_truth)
            total_fp += result["fp"]
            for error in result["fp_details"]:
                counts[error["type"]] += 1
                index = error["best_index"]
                target = f"UAV#{index + 1}" if index is not None else "none"
                # retain the overlap so borderline categories can be inspected.
                print(
                    f"{label} {name}: {error['type']}, score={error['score']:.3f}, "
                    f"best-overlap target={target}, IoU={error['best_iou']:.4f}"
                )
        assert sum(counts.values()) == total_fp, f"{label}: error counts do not add up to FP"
        error_summaries[label] = counts

    print("\nMethod | Duplicate | Localisation candidate | Background candidate")
    for label, counts in error_summaries.items():
        print(
            f"{label} | {counts['duplicate']} | "
            f"{counts['localisation_candidate']} | {counts['background_candidate']}"
        )

    print("\n=== IoU SENSITIVITY ===")
    for threshold in (0.3, 0.5, 0.75):
        print(f"\nIoU threshold: {threshold}")
        for label, image_predictions in prediction_cache.items():
            totals = {"tp": 0, "fp": 0, "fn": 0}
            for name, ground_truth in annotations.items():
                result = match_predictions(image_predictions[name], ground_truth, threshold)
                for key in totals:
                    totals[key] += result[key]
            tp, fp, fn = totals.values()
            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            print(f"{label}: Recall={recall:.1%}, Precision={precision:.1%}, FP={fp}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=CACHE_PATH, help="saved predictions to inspect"
    )
    args = parser.parse_args()
    data = json.loads(args.cache.read_text(encoding="utf-8"))
    if not data["annotations"]:
        parser.error("the cache contains no images")
    if "image_sizes" not in data:
        parser.error("size analysis needs image_sizes in the cache; use the published snapshot")
    analyse(data)


if __name__ == "__main__":
    main()
