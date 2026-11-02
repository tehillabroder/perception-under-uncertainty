"""
Add SAHI with full-image inference to a copy of the published post1 cache
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from perception_under_uncertainty.evaluation import IOU_THRESHOLD, match_predictions
from scripts.evaluate_post1 import (
    CONF_THRESHOLD,
    IMAGES_DIR,
    MODEL_FILENAME,
    MODEL_REPO,
    SAHI_SETTINGS,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/post1_predictions_with_full_image_rerun.json"),
        help="destination for a new run; the frozen exploratory cache is kept separately",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new --output path")
    baseline = json.loads(Path("results/post1/predictions.json").read_text(encoding="utf-8"))
    image_names = baseline["predictions"]["SAHI 640"]
    assert len(image_names) == 30
    assert set(image_names) == set(baseline["annotations"])
    assert baseline["model"] == MODEL_REPO
    assert baseline["confidence_threshold"] == CONF_THRESHOLD == 0.25
    assert baseline["inference_settings"]["sahi"] == SAHI_SETTINGS
    settings = {**SAHI_SETTINGS, "perform_standard_pred": True}
    weights = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_files_only=True)
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=CONF_THRESHOLD,
        image_size=640,
        device="cpu",
    )

    new_predictions = {}
    for name in image_names:
        result = get_sliced_prediction(str(IMAGES_DIR / name), model, **settings, verbose=0)
        new_predictions[name] = [
            {
                "box": pred.bbox.to_xyxy(),
                "score": float(pred.score.value),
                "label": pred.category.name,
            }
            for pred in result.object_prediction_list
        ]
        print(f"{name}: {len(new_predictions[name])} predictions", flush=True)

    baseline["predictions"]["SAHI 640 + full image"] = new_predictions
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved predictions to {output}")

    print(f"\nConfidence: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD}")
    print("Method | TP | FP | FN | Recall | Precision")
    for label, predictions in baseline["predictions"].items():
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for name, annotations in baseline["annotations"].items():
            result = match_predictions(predictions[name], annotations)
            for key in totals:
                totals[key] += result[key]
        tp, fp, fn = totals.values()
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        print(f"{label} | {tp} | {fp} | {fn} | {recall:.2%} | {precision:.2%}")


if __name__ == "__main__":
    main()
