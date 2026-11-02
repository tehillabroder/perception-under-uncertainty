"""Render five DUT evaluation cases from saved predictions, without inference.

Run from the repository root:
    uv run --locked python -m scripts.visualise_post1_cases
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from perception_under_uncertainty.evaluation import IOU_THRESHOLD, box_iou, match_predictions
from scripts.make_post1_visuals import BROWN, CHARCOAL, CREAM, INK, MONO, RUST, SANS, SERIF, TAUPE

CACHE_PATH = Path("runs/dut_300_evaluation/predictions.json")
IMAGES_DIR = Path("data/raw/dut_test/img")
CASES = ("00395.jpg", "00150.jpg", "00224.jpg", "00622.jpg", "00787.jpg")
METHODS = ("YOLO 640", "YOLO 1280", "SAHI 640", "SAHI 640 + full image")


def target_view(annotations, width, height):
    """Use the same annotation-derived crop for every method, in original pixels."""
    if not annotations:
        return (0, 0, width, height)
    boxes = [annotation["box"] for annotation in annotations]
    x1, y1 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x2, y2 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    crop_width, crop_height = max(96, 2 * (x2 - x1)), max(96, 2 * (y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        max(0, cx - crop_width / 2),
        max(0, cy - crop_height / 2),
        min(width, cx + crop_width / 2),
        min(height, cy + crop_height / 2),
    )


def draw_box(ax, box, colour, dashed=False):
    x1, y1, x2, y2 = box
    patch = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=colour,
        linewidth=1.5,
        linestyle="--" if dashed else "-",
    )
    patch.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground=INK), path_effects.Normal()]
    )
    ax.add_patch(patch)


def draw_panel(ax, image, annotations, predictions, view):
    ax.imshow(image, interpolation="nearest")
    left, top, right, bottom = view
    ax.set_xlim(left - 0.5, right - 0.5)
    ax.set_ylim(bottom - 0.5, top - 0.5)
    ax.set_axis_off()
    for annotation in annotations:
        draw_box(ax, annotation["box"], TAUPE, dashed=True)

    # Keep boxes unobstructed; confidence and IoU appear in the panel headings.
    for prediction in predictions:
        draw_box(ax, prediction["box"], RUST)


def render_case(data, name, images_dir, output):
    with Image.open(images_dir / name) as source:
        image = source.convert("RGB")
    if image.size != tuple(data["image_sizes"][name]):
        raise ValueError(f"Image dimensions differ from the prediction cache: {name}")
    width, height = image.size
    annotations = data["annotations"][name]
    # Localisation cases use a target crop; background errors need the full scene.
    view = (0, 0, width, height) if name == "00787.jpg" else target_view(annotations, width, height)
    fig, axes = plt.subplots(1, 5, figsize=(19, 5.8), dpi=160, facecolor=CREAM)
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.14, top=0.73, wspace=0.08)
    fig.suptitle(f"DUT case study / {name}", fontfamily=SERIF, fontsize=26, color=INK, y=0.97)
    fig.text(
        0.025,
        0.89,
        f"Saved predictions | confidence >= {data['confidence_threshold']:.2f} | "
        f"matching IoU >= {IOU_THRESHOLD:.2f} | original image {width} x {height} px",
        fontfamily=MONO,
        fontsize=10,
        color=BROWN,
    )
    for column, method in enumerate(("Ground truth", *METHODS)):
        predictions = (
            []
            if column == 0
            else sorted(data["predictions"][method][name], key=lambda p: p["score"], reverse=True)
        )
        if column == 0:
            detail = f"{len(annotations)} annotated UAV(s)"
        else:
            result = match_predictions(predictions, annotations)
            detail = f"TP {result['tp']} / FP {result['fp']} / FN {result['fn']}"
            print(f"{name} | {method} | {detail}")
        scores = []
        for prediction in predictions:
            overlap = max((box_iou(prediction["box"], a["box"]) for a in annotations), default=0.0)
            scores.append(f"confidence {prediction['score']:.2f} / IoU {overlap:.2f}")
        title = "\n".join((method, detail, *scores))
        draw_panel(axes[column], image, annotations, predictions, view)
        axes[column].set_title(title, fontfamily=SANS, fontsize=11, color=INK, pad=12)

    fig.text(
        0.025,
        0.065,
        "Dashed taupe: ground truth. Solid rust: prediction. Counts refer to the full image; "
        "a target crop can hide off-target false positives.",
        fontfamily=SANS,
        fontsize=11,
        color=CHARCOAL,
    )
    fig.text(
        0.025,
        0.035,
        f"{'Full scene' if name == '00787.jpg' else 'Target-centred zoom'}. "
        "Headings list all predictions by descending confidence; IoU is the best overlap "
        "with any annotation.",
        fontfamily=SANS,
        fontsize=11,
        color=CHARCOAL,
    )
    filename = output / f"{Path(name).stem}_comparison.png"
    fig.savefig(filename, facecolor=CREAM)
    plt.close(fig)
    print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, help="defaults to visuals/ beside the cache")
    parser.add_argument("--images", nargs="+", default=CASES, help="image basenames to render")
    args = parser.parse_args()
    data = json.loads(args.cache.read_text(encoding="utf-8"))
    for name in args.images:
        if Path(name).name != name or name not in data["annotations"]:
            parser.error(f"Image is not a basename in the saved annotations: {name}")
        if not (args.images_dir / name).is_file():
            parser.error(f"Missing source image: {args.images_dir / name}")
        for method in METHODS:
            if name not in data["predictions"].get(method, {}):
                parser.error(f"Missing saved predictions for {method}: {name}")
    output = args.output_dir or args.cache.parent / "visuals"
    output.mkdir(parents=True, exist_ok=True)
    for name in args.images:
        render_case(data, name, args.images_dir, output)


if __name__ == "__main__":
    main()
