"""plot post1 false-positive groups from saved predictions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from perception_under_uncertainty.evaluation import IOU_THRESHOLD, match_predictions
from scripts.make_post1_visuals import (
    BROWN,
    CACHE_PATH,
    CHARCOAL,
    CREAM,
    FIGURES_DIR,
    INK,
    MONO,
    RUST,
    SANS,
    SERIF,
    TAUPE,
)

COLOURS = (TAUPE, RUST, BROWN)


def load_error_counts(data):
    prediction_cache = data["predictions"]
    annotations = data["annotations"]

    results = []

    for label in ("YOLO 640", "YOLO 1280", "SAHI 640"):
        counts = {
            "duplicate": 0,
            "localisation_candidate": 0,
            "background_candidate": 0,
        }

        for image_name, predictions in prediction_cache[label].items():
            result = match_predictions(
                predictions,
                annotations[image_name],
                IOU_THRESHOLD,
            )
            for error in result["fp_details"]:
                counts[error["type"]] += 1

        total = sum(counts.values())

        results.append(
            (
                label,
                counts["duplicate"],
                counts["localisation_candidate"],
                counts["background_candidate"],
                total,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.cache.read_text(encoding="utf-8"))
    n_images = len(data["annotations"])
    results = load_error_counts(data)

    # use the measured count in the callout text.
    sahi_row = next(row for row in results if row[0] == "SAHI 640")
    _, _, _, sahi_low_overlap, sahi_total = sahi_row

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
        0.875,
        "What did slicing\nactually find?",
        fontfamily=SERIF,
        fontsize=43,
        color=INK,
        va="top",
        linespacing=1.13,
    )
    fig.text(
        0.08,
        0.710,
        "A closer look at the predictions that failed matching on",
        fontfamily=SANS,
        fontsize=13,
        color=CHARCOAL,
    )
    fig.text(
        0.08,
        0.685,
        f"{n_images} selected DUT Anti-UAV images.",
        fontfamily=SANS,
        fontsize=13,
        color=CHARCOAL,
    )

    fig.add_artist(
        Line2D(
            [0.08, 0.92],
            [0.649, 0.649],
            transform=fig.transFigure,
            color=TAUPE,
            alpha=0.6,
            linewidth=0.8,
        )
    )

    fig.text(
        0.08,
        0.612,
        "FALSE POSITIVES BY GROUP",
        fontfamily=MONO,
        fontsize=11,
        color=BROWN,
    )

    for x, name, colour in [
        (0.08, "Duplicates", TAUPE),
        (0.32, "Localisation", RUST),
        (0.60, "Low overlap", BROWN),
    ]:
        fig.add_artist(
            Line2D(
                [x, x + 0.025],
                [0.570, 0.570],
                transform=fig.transFigure,
                color=colour,
                linewidth=8,
                solid_capstyle="butt",
            )
        )
        fig.text(
            x + 0.035,
            0.563,
            name,
            fontfamily=SANS,
            fontsize=11,
            color=CHARCOAL,
        )

    ax = fig.add_axes((0.255, 0.343, 0.655, 0.195))

    max_total = max(row[4] for row in results)
    x_limit = max_total + 5

    for position, (_method, duplicates, localisation, low_overlap, total) in zip(
        [2, 1, 0], results, strict=True
    ):
        left = 0

        for count, colour in zip((duplicates, localisation, low_overlap), COLOURS, strict=True):
            if count:
                ax.barh(
                    position,
                    count,
                    left=left,
                    height=0.52,
                    color=colour,
                    edgecolor=CREAM,
                    linewidth=1,
                    zorder=2,
                )
                ax.text(
                    left + count / 2,
                    position,
                    str(count),
                    ha="center",
                    va="center",
                    fontfamily=MONO,
                    fontsize=9 if count <= 2 else 11,
                    color=INK if colour == TAUPE else CREAM,
                    zorder=3,
                )
            left += count

        ax.text(
            total + 0.7,
            position,
            str(total),
            ha="left",
            va="center",
            fontfamily=MONO,
            fontsize=12,
            color=INK,
        )

    ax.set_facecolor(CREAM)
    ax.set_yticks([2, 1, 0], [row[0] for row in results])
    ax.tick_params(
        axis="y",
        length=0,
        labelsize=13,
        pad=12,
        colors=CHARCOAL,
    )
    ax.set_xlim(0, x_limit)
    ax.set_ylim(-0.55, 2.55)

    tick_step = 10
    ticks = list(range(0, int(x_limit) + 1, tick_step))
    ax.set_xticks(ticks)
    ax.tick_params(
        axis="x",
        length=0,
        labelsize=10,
        pad=12,
        colors=TAUPE,
    )
    ax.grid(axis="x", color=TAUPE, alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.add_artist(
        Line2D(
            [0.08, 0.92],
            [0.285, 0.285],
            transform=fig.transFigure,
            color=TAUPE,
            alpha=0.6,
            linewidth=0.8,
        )
    )

    fig.text(
        0.08,
        0.203,
        f"{sahi_low_overlap} / {sahi_total}",
        fontfamily=SERIF,
        fontsize=43,
        color=BROWN,
    )
    fig.text(
        0.40,
        0.229,
        "SAHI false positives were",
        fontfamily=SANS,
        fontsize=13,
        color=CHARCOAL,
    )
    fig.text(
        0.40,
        0.203,
        "low-overlap candidates.",
        fontfamily=SANS,
        fontsize=13,
        color=CHARCOAL,
    )

    fig.add_artist(
        Line2D(
            [0.08, 0.92],
            [0.165, 0.165],
            transform=fig.transFigure,
            color=TAUPE,
            alpha=0.6,
            linewidth=0.8,
        )
    )
    fig.text(
        0.08,
        0.130,
        "Low overlap does not necessarily mean verified background confusion.",
        fontfamily=SANS,
        fontsize=10,
        color=CHARCOAL,
    )
    fig.text(
        0.08,
        0.103,
        "Categories describe unmatched predictions, not unique missed UAVs.",
        fontfamily=SANS,
        fontsize=10,
        color=CHARCOAL,
    )
    fig.text(
        0.08,
        0.070,
        f"Confidence {data['confidence_threshold']}  |  IoU {IOU_THRESHOLD:.2f}  |  "
        f"Exploratory sample of {n_images} images.",
        fontfamily=MONO,
        fontsize=9,
        color=BROWN,
    )

    filename = output / "post1_error_breakdown.png"
    fig.savefig(filename, dpi=135, facecolor=CREAM)
    plt.close(fig)

    print(f"Saved {filename}")


if __name__ == "__main__":
    main()
