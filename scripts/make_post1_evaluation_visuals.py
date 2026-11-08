"""Compare drones found and extra boxes on one slide for the four-method DUT evaluation.

Run from the repository root:
    uv run --locked python -m scripts.make_post1_evaluation_visuals
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from perception_under_uncertainty.evaluation import IOU_THRESHOLD, match_predictions
from scripts.make_post1_visuals import (
    BROWN,
    CHARCOAL,
    CREAM,
    INK,
    MONO,
    SANS,
    SERIF,
    TAUPE,
    bar_panel,
)

CACHE_PATH = Path("runs/dut_300_evaluation/predictions.json")
METHODS = ("YOLO 640", "YOLO 1280", "SAHI 640", "SAHI 640 + full image")
LABELS = ("YOLO 640", "YOLO 1280", "SAHI 640", "SAHI 640\n+ full image")


def evaluation_totals(data):
    """Reapply the case-study matching rule to every image, including empty predictions."""
    annotations = data["annotations"]
    if not annotations or not any(annotations.values()):
        raise ValueError("The evaluation needs images and annotated drones.")
    results = []
    for method in METHODS:
        predictions = data["predictions"][method]
        if predictions.keys() != annotations.keys():
            raise ValueError(f"Saved predictions and annotations cover different images: {method}")
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for name, ground_truth in annotations.items():
            result = match_predictions(predictions[name], ground_truth)
            for key in totals:
                totals[key] += result[key]
        results.append(totals)
    return results


def render_slide(data, output, recall, fp_per_image):
    n_images = len(data["annotations"])
    n_objects = sum(len(boxes) for boxes in data["annotations"].values())
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=CREAM)
    fig.text(
        0.05, 0.95, "PERCEPTION UNDER UNCERTAINTY",
        fontfamily=MONO, fontsize=10, color=BROWN,
    )
    for left, title, description in (
        (
            0.20, "Which method found\nmore drones?",
            "Share of labelled drones\ncorrectly detected",
        ),
        (
            0.61, "Which method produced\nmore extra boxes?",
            "Extra detections, including inaccurate\nand duplicate boxes (per image)",
        ),
    ):
        fig.text(
            left, 0.89, title, fontfamily=SERIF, fontsize=27, color=INK,
            va="top", linespacing=1.05,
        )
        fig.text(
            left, 0.77, description, fontfamily=SANS, fontsize=12,
            color=CHARCOAL, va="top", linespacing=1.3,
        )

    ax_recall = fig.add_axes((0.20, 0.22, 0.32, 0.48))
    bar_panel(
        ax_recall, LABELS, recall, 100, [0, 25, 50, 75, 100],
        lambda value: f"{value:.1f}%",
    )
    ax_recall.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontfamily=MONO)

    # Counts per image can exceed one; they are not percentages.
    fp_maximum = max(1.5, math.ceil(max(fp_per_image) * 1.2 * 2) / 2)
    fp_ticks = [index * 0.5 for index in range(round(fp_maximum * 2) + 1)]
    ax_fp = fig.add_axes((0.61, 0.22, 0.33, 0.48))
    bar_panel(
        ax_fp, LABELS, fp_per_image, fp_maximum, fp_ticks,
        lambda value: f"{value:.2f}",
    )
    ax_fp.set_xticklabels([f"{value:g}" for value in fp_ticks], fontfamily=MONO)
    # The two panels share method rows and colours, so label each row just once.
    ax_fp.tick_params(axis="y", labelleft=False)

    winners = [
        method for method, value in zip(METHODS, recall, strict=True) if value == max(recall)
    ]
    takeaway = f"{' and '.join(winners)} found the most drones."
    fig.text(0.05, 0.105, takeaway, fontfamily=SERIF, fontsize=23, color=BROWN)
    fig.add_artist(
        Line2D(
            [0.05, 0.95], [0.075, 0.075], transform=fig.transFigure,
            color=TAUPE, alpha=0.6, linewidth=0.8,
        )
    )
    fig.text(
        0.05, 0.04,
        f"DUT Anti-UAV  |  {n_images} images  |  {n_objects} labelled drones  |  "
        f"Same pretrained model  |  Confidence >= {data['confidence_threshold']:.2f}  |  "
        f"Matching IoU >= {IOU_THRESHOLD:.2f}",
        fontfamily=MONO, fontsize=9, color=CHARCOAL,
    )
    path = output / "post1_evaluation_comparison.png"
    fig.savefig(path, facecolor=CREAM)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, help="defaults to visuals/ beside the cache")
    args = parser.parse_args()
    data = json.loads(args.cache.read_text(encoding="utf-8"))
    results = evaluation_totals(data)
    n_images = len(data["annotations"])
    recall = [100 * row["tp"] / (row["tp"] + row["fn"]) for row in results]
    fp_per_image = [row["fp"] / n_images for row in results]
    output = args.output_dir or args.cache.parent / "visuals"
    output.mkdir(parents=True, exist_ok=True)

    render_slide(data, output, recall, fp_per_image)
    for method, row, recall_value, fp_value in zip(
        METHODS, results, recall, fp_per_image, strict=True
    ):
        print(
            f"{method}: TP {row['tp']} / FP {row['fp']} / FN {row['fn']} | "
            f"recall {recall_value:.1f}% | false positives/image {fp_value:.2f}"
        )


if __name__ == "__main__":
    main()
