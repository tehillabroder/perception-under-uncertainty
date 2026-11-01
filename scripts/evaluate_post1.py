from pathlib import Path
from xml.etree import ElementTree as ET

from huggingface_hub import hf_hub_download
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

IMAGES_DIR = Path("data/post1_sample")
XML_DIR = Path("data/raw/dut_test/xml")

# Pretrained UAV detector from Hugging Face
MODEL_REPO = "TomSmail/drone-yolo-v1"
MODEL_FILENAME = "best.pt"

METHODS = (
    ("YOLO", 640),
    ("YOLO", 1280),
    ("SAHI", 640),
)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5


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
        annotations.append(
            {
                "label": obj.findtext("name"),
                "box": coordinates,
            }
        )

    return annotations


def box_iou(a, b):
    overlap_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))

    intersection = overlap_width * overlap_height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    if area_a <= 0 or area_b <= 0:
        raise ValueError("Bounding boxes must have positive area")
    union = area_a + area_b - intersection

    return intersection / union


def prediction_score(prediction):
    return prediction["score"]


def match_predictions(predictions, annotations, iou_threshold=IOU_THRESHOLD):
    """
    Sort predictions by confidence score descending before matching.
    Matches each prediction to its best-overlapping ground-truth box.
    If the best-matching annotation was already claimed by a higher-confidence
    prediction, this detection is counted as a false positive (duplicate).
    """
    predictions = sorted(predictions, key=prediction_score, reverse=True)
    matched_annotations = set()
    true_positives = 0
    false_positives = 0

    for prediction in predictions:
        best_index = None
        best_iou = 0

        for index, annotation in enumerate(annotations):
            overlap = box_iou(prediction["box"], annotation["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if (
            best_index is not None
            and best_iou >= iou_threshold
            and best_index not in matched_annotations
        ):
            matched_annotations.add(best_index)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(annotations) - len(matched_annotations)

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "matched_indices": matched_annotations,
    }


def predict_yolo(model, image_path, image_size):
    """
    Run the same pretrained model on one image.
    Extract every predicted bounding box in original-image coordinates.
    Keep its confidence score and class label.
    Return a list in the format that the matching function expects.
    """
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
    result = get_sliced_prediction(
        str(image_path),
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,
        verbose=0,
    )

    predictions = []
    for pred in result.object_prediction_list:
        predictions.append(
            {
                "box": pred.bbox.to_xyxy(),
                "score": float(pred.score.value),
                "label": pred.category.name,
            }
        )

    return predictions


def main():
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {IMAGES_DIR}")

    annotations = {image.name: read_annotations(XML_DIR / f"{image.stem}.xml") for image in images}

    # approximate object size after whole-image resizing to 640
    scales = {}
    for image in images:
        with Image.open(image) as img:
            width, height = img.size
        scales[image.name] = min(640 / width, 640 / height)

    weights = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
    )

    model = YOLO(weights)
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=CONF_THRESHOLD,
        image_size=640,
        device="cpu",
    )

    matches = {}
    prediction_cache = {}  # store predictions to avoid re-running inference

    print(f"Model: {MODEL_REPO}/{MODEL_FILENAME}")
    print(f"Images: {len(images)}")
    print(f"Confidence: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD}")

    for method, image_size in METHODS:
        label = f"{method} {image_size}"
        matches[label] = {}
        prediction_cache[label] = {}  # Added
        totals = {"tp": 0, "fp": 0, "fn": 0}

        print(f"\n--- {label}: Missed Targets Analysis ---")
        for image in images:
            if method == "SAHI":
                predictions = predict_sahi(sahi_model, image)
            else:
                predictions = predict_yolo(model, image, image_size)

            prediction_cache[label][image.name] = predictions

            result = match_predictions(predictions, annotations[image.name])
            matches[label][image.name] = result["matched_indices"]

            for key in totals:
                totals[key] += result[key]

            # Audit unmatched annotations for near-misses vs true misses
            for index, annotation in enumerate(annotations[image.name]):
                if index not in result["matched_indices"]:
                    best_iou = max(
                        (box_iou(p["box"], annotation["box"]) for p in predictions),
                        default=0.0,
                    )
                    print(f"{label} {image.name} UAV#{index + 1}: best IoU = {best_iou:.3f}")

        tp, fp, fn = totals.values()

        print(f"\n{label} SUMMARY")
        print(f"TP={tp}, FP={fp}, FN={fn}")
        print(f"Recall={tp / (tp + fn):.2%}")
        print(f"Precision={tp / (tp + fp):.2%}")
        print(f"FP/image={fp / len(images):.2f}")

    def size_at_640(image_name, index):
        x1, y1, x2, y2 = annotations[image_name][index]["box"]
        return ((x2 - x1) * (y2 - y1)) ** 0.5 * scales[image_name]

    # which exact UAVs are gained or lost?
    print("\n=== GAINED / LOST vs YOLO 640 ===")
    baseline = matches["YOLO 640"]

    for label in ("YOLO 1280", "SAHI 640"):
        print(f"\n{label}")
        for image in images:
            name = image.name
            gained = matches[label][name] - baseline[name]
            lost = baseline[name] - matches[label][name]
            for index in sorted(gained):
                print(f"GAIN {name} UAV#{index + 1}: {size_at_640(name, index):.1f}px")
            for index in sorted(lost):
                print(f"LOST {name} UAV#{index + 1}: {size_at_640(name, index):.1f}px")

    # does recall depend on object size?
    print("\n=== RECALL BY UAV SIZE ===")
    bins = (
        ("<16", 0, 16),
        ("16-31", 16, 32),
        ("32-63", 32, 64),
        ("64+", 64, float("inf")),
    )

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

    # iou sensitivity sweep across cached predictions
    print("\n=== IoU SENSITIVITY ===")
    for threshold in (0.3, 0.5, 0.75):
        print(f"\nIoU threshold: {threshold}")
        for label, image_predictions in prediction_cache.items():
            totals = {"tp": 0, "fp": 0, "fn": 0}
            for image in images:
                result = match_predictions(
                    image_predictions[image.name],
                    annotations[image.name],
                    threshold,
                )
                for key in totals:
                    totals[key] += result[key]
            tp, fp, fn = totals.values()
            print(f"{label}: Recall={tp / (tp + fn):.1%}, Precision={tp / (tp + fp):.1%}, FP={fp}")


if __name__ == "__main__":
    main()
