"""
Run the post1 baselines and save predictions for later analysis
"""

import argparse
import json
import platform
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree as ET

from huggingface_hub import hf_hub_download
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from perception_under_uncertainty import evaluation
from perception_under_uncertainty.evaluation import IOU_THRESHOLD, match_predictions

IMAGES_DIR = Path("data/post1_sample")
XML_DIR = Path("data/raw/dut_test/xml")
CACHE_PATH = Path("runs/post1_predictions.json")
MODEL_REPO = "TomSmail/drone-yolo-v1"
MODEL_FILENAME = "best.pt"
METHODS = (("YOLO", 640), ("YOLO", 1280), ("SAHI", 640))
CONF_THRESHOLD = 0.25
SAHI_SETTINGS = {
    "slice_height": 640,
    "slice_width": 640,
    "overlap_height_ratio": 0.2,
    "overlap_width_ratio": 0.2,
    "perform_standard_pred": False,
    "postprocess_type": "GREEDYNMM",
    "postprocess_match_metric": "IOS",
    "postprocess_match_threshold": 0.5,
}


def read_annotations(xml_path):
    root = ET.parse(xml_path).getroot()
    annotations = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {xml_path}")
        coordinates = []
        for key in ("xmin", "ymin", "xmax", "ymax"):
            value = box.findtext(key)
            if value is None:
                raise ValueError(f"Missing {key} in {xml_path}")
            coordinates.append(float(value))
        annotations.append({"label": obj.findtext("name"), "box": coordinates})
    return annotations


def predict_yolo(model, image_path, image_size):
    """keep boxes in original-image coordinates, with scores and labels."""
    result = model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=CONF_THRESHOLD,
        device="cpu",
        save=False,
        verbose=False,
    )[0]
    predictions = []
    if result.boxes is None:
        return predictions
    for detected_box in result.boxes:
        predictions.append(
            {
                "box": detected_box.xyxy[0].tolist(),
                "score": float(detected_box.conf.item()),
                "label": model.names[int(detected_box.cls.item())],
            }
        )
    return predictions


def predict_sahi(model, image_path):
    result = get_sliced_prediction(str(image_path), model, **SAHI_SETTINGS, verbose=0)
    return [
        {
            "box": pred.bbox.to_xyxy(),
            "score": float(pred.score.value),
            "label": pred.category.name,
        }
        for pred in result.object_prediction_list
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=CACHE_PATH, help="destination prediction cache"
    )
    args = parser.parse_args()
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {IMAGES_DIR}")

    annotations = {image.name: read_annotations(XML_DIR / f"{image.stem}.xml") for image in images}
    image_sizes = {}
    for image in images:
        with Image.open(image) as img:
            image_sizes[image.name] = list(img.size)

    weights = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = YOLO(weights)
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=CONF_THRESHOLD,
        image_size=640,
        device="cpu",
    )
    prediction_cache = {}
    started_at = datetime.now(UTC).isoformat()
    print(f"Model: {MODEL_REPO}/{MODEL_FILENAME}")
    print(f"Images: {len(images)}")
    print(f"Confidence: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD}")

    for method, image_size in METHODS:
        label = f"{method} {image_size}"
        prediction_cache[label] = {}
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for image in images:
            if method == "SAHI":
                predictions = predict_sahi(sahi_model, image)
            else:
                predictions = predict_yolo(model, image, image_size)
            prediction_cache[label][image.name] = predictions
            result = match_predictions(predictions, annotations[image.name])
            for key in totals:
                totals[key] += result[key]

        tp, fp, fn = totals.values()
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        print(f"\n{label}: TP={tp}, FP={fp}, FN={fn}")
        print(f"Recall={recall:.2%}, Precision={precision:.2%}, FP/image={fp / len(images):.2f}")

    # save enough context to analyse the run without reopening the images.
    cache = {
        "model": MODEL_REPO,
        "confidence_threshold": CONF_THRESHOLD,
        "predictions": prediction_cache,
        "annotations": annotations,
        "image_sizes": image_sizes,
        "inference_settings": {
            "device": "cpu",
            "yolo_image_sizes": [640, 1280],
            "sahi_image_size": 640,
            "sahi": SAHI_SETTINGS,
        },
        "provenance": {
            "started_at_utc": started_at,
            "model_filename": MODEL_FILENAME,
            "weights_sha256": sha256(Path(weights).read_bytes()).hexdigest(),
            "source_sha256": {
                "evaluate_post1.py": sha256(Path(__file__).read_bytes()).hexdigest(),
                "evaluation.py": sha256(Path(evaluation.__file__).read_bytes()).hexdigest(),
            },
            "python_version": platform.python_version(),
            "packages": {
                name: version(name) for name in ("ultralytics", "sahi", "torch", "torchvision")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
