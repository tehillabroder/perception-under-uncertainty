"""plot the post1 baseline comparison from saved predictions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from perception_under_uncertainty.evaluation import IOU_THRESHOLD, match_predictions

CACHE_PATH = Path("results/post1/predictions.json")
FIGURES_DIR = Path("results/post1/figures")

# shared colours for the post1 figures
CREAM = "#F4F0E8"
INK = "#11100F"
CHARCOAL = "#24211F"
BROWN = "#684131"
TAUPE = "#A99B8E"
RUST = "#98695E"

METHOD_COLOURS = [TAUPE, BROWN, RUST]


def available_font(*choices):
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next(
        (font for font in choices if font in installed),
        "DejaVu Sans",
    )


SERIF = available_font("Instrument Serif", "Newsreader", "Noto Serif Display", "DejaVu Serif")
SANS = available_font("IBM Plex Sans", "Inter", "Lato", "DejaVu Sans")
MONO = available_font("IBM Plex Mono", "DejaVu Sans Mono")


def bar_panel(ax, labels, values, maximum, ticks, format_value):
    positions = [2, 1, 0]

    ax.barh(
        positions,
        values,
        height=0.46,
        color=METHOD_COLOURS,
        zorder=2,
    )

    ax.set_yticks(
        positions,
        labels=labels,
        fontsize=13,
        fontfamily=SANS,
        color=CHARCOAL,
    )
    ax.set_xlim(0, maximum)
    ax.set_ylim(-0.65, 2.65)
    ax.set_xticks(ticks)
    ax.tick_params(
        axis="x",
        length=0,
        labelsize=10,
        colors=TAUPE,
        pad=10,
    )
    ax.tick_params(axis="y", length=0, pad=12)

    for tick in ticks:
        ax.axvline(
            tick,
            color=TAUPE,
            alpha=0.3,
            linewidth=0.8,
            zorder=0,
        )

    for pos, value in zip(positions, values, strict=True):
        ax.text(
            value + maximum * 0.025,
            pos,
            format_value(value),
            va="center",
            ha="left",
            fontfamily=MONO,
            fontsize=12,
            color=INK,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_facecolor(CREAM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    data = json.loads(args.cache.read_text(encoding="utf-8"))
    annotations = data["annotations"]
    n_images = len(annotations)
    n_objects = sum(len(boxes) for boxes in annotations.values())
    results = []
    for method in ("YOLO 640", "YOLO 1280", "SAHI 640"):
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for name, ground_truth in annotations.items():
            result = match_predictions(data["predictions"][method][name], ground_truth)
            for key in totals:
                totals[key] += result[key]
        results.append((method, totals["tp"], totals["fp"], totals["fn"]))
    labels = [row[0] for row in results]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 10), dpi=135, facecolor=CREAM)

    fig.text(
        0.08,
        0.945,
        "PERCEPTION UNDER UNCERTAINTY",
        fontfamily=MONO,
        fontsize=10,
        color=BROWN,
    )

    fig.text(
        0.08,
        0.877,
        "More detail.\nMore false alarms?",
        fontfamily=SERIF,
        fontsize=43,
        color=INK,
        va="top",
        linespacing=1.13,
    )

    fig.text(
        0.08,
        0.715,
        "A comparison of whole-image and sliced inference on",
        fontfamily=SANS,
        fontsize=13,
        color=CHARCOAL,
    )
    fig.text(
        0.08,
        0.690,
        f"{n_images} selected DUT Anti-UAV images, ranging in difficulty and size.",
        fontfamily=SANS,
        fontsize=13,
        color=CHARCOAL,
    )

    fig.add_artist(
        Line2D(
            [0.08, 0.92],
            [0.653, 0.653],
            transform=fig.transFigure,
            color=TAUPE,
            alpha=0.6,
            linewidth=0.8,
        )
    )

    # recall at the fixed matching threshold
    fig.text(
        0.08,
        0.613,
        "UAV RECALL",
        fontfamily=MONO,
        fontsize=11,
        color=BROWN,
    )
    fig.text(
        0.08,
        0.588,
        f"Matched detections / {n_objects} annotated UAVs",
        fontfamily=SANS,
        fontsize=11,
        color=CHARCOAL,
    )

    ax_recall = fig.add_axes((0.265, 0.415, 0.615, 0.14))
    recall = [100 * tp / (tp + fn) for _, tp, _, fn in results]

    bar_panel(
        ax_recall,
        labels,
        recall,
        100,
        [0, 25, 50, 75, 100],
        lambda value: f"{value:.1f}%",
    )
    ax_recall.set_xticklabels(["0", "25", "50", "75", "100%"])

    # include every unmatched prediction in the false-positive count
    fig.text(
        0.08,
        0.353,
        "FALSE POSITIVES / IMAGE",
        fontfamily=MONO,
        fontsize=11,
        color=BROWN,
    )
    fig.text(
        0.08,
        0.328,
        "Includes duplicate detections",
        fontfamily=SANS,
        fontsize=11,
        color=CHARCOAL,
    )

    ax_fp = fig.add_axes((0.265, 0.16, 0.615, 0.14))
    fp_per_image = [fp / n_images for _, _, fp, _ in results]

    bar_panel(
        ax_fp,
        labels,
        fp_per_image,
        max(1.55, max(fp_per_image) * 1.2),
        [0, 0.5, 1.0, 1.5],
        lambda value: f"{value:.2f}",
    )
    ax_fp.set_xticklabels(["0", "0.5", "1.0", "1.5"])

    # keep the settings visible alongside the measurements
    fig.add_artist(
        Line2D(
            [0.08, 0.92],
            [0.115, 0.115],
            transform=fig.transFigure,
            color=TAUPE,
            alpha=0.6,
            linewidth=0.8,
        )
    )

    fig.text(
        0.08,
        0.092,
        f"Same pretrained model  |  confidence {data['confidence_threshold']}  |  "
        f"IoU {IOU_THRESHOLD:.2f}",
        fontfamily=MONO,
        fontsize=9,
        color=CHARCOAL,
    )
    fig.text(
        0.08,
        0.065,
        f"Exploratory sample of {n_images} images.",
        fontfamily=SANS,
        fontsize=10,
        color=BROWN,
    )

    filename = output / "post1_method_comparison.png"
    fig.savefig(filename, dpi=135, facecolor=CREAM)
    plt.close(fig)

    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
