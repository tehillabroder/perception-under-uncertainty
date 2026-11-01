from pathlib import Path
from xml.etree import ElementTree as ET

from huggingface_hub import hf_hub_download
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


def match_predictions(predictions, annotations):
    """
    My plan is sorting by confidence before matching.
    If the model predicts two boxes around the same drone, only one should count as correct.
    This stops duplicate detections from artificially increasing recall.
    For now, i'm counting duplicates as false positives.
    Later I can distinguish background false positives from duplicates for failure analysis.
    """
    predictions = sorted(predictions, key=prediction_score, reverse=True)
    matched_annotations = set()
    true_positives = 0
    false_positives = 0
    for prediction in predictions:
        best_index = None
        best_iou = 0
        for index, annotation in enumerate(annotations):
            if index in matched_annotations:
                continue
            overlap = box_iou(prediction["box"], annotation["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= IOU_THRESHOLD:
            matched_annotations.add(best_index)
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(annotations) - len(matched_annotations)

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
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

    # download pretrained weights from Hugging Face (cached locally)
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

    print(f"Model: {MODEL_REPO}/{MODEL_FILENAME}")
    print(f"Images: {len(images)}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

    for method, image_size in METHODS:
        label = f"{method} {image_size}"
        totals = {"tp": 0, "fp": 0, "fn": 0}
        print(f"\n=== {label} ===")

        for image in images:
            xml_path = XML_DIR / f"{image.stem}.xml"
            annotations = read_annotations(xml_path)
            if method == "SAHI":
                predictions = predict_sahi(sahi_model, image)
            else:
                predictions = predict_yolo(model, image, image_size)
            results = match_predictions(predictions, annotations)

            print(f"{image.name}: {results}")

            for key in totals:
                totals[key] += results[key]

        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]

        recall = tp / (tp + fn) if tp + fn else 0
        precision = tp / (tp + fp) if tp + fp else 0

        print(f"\n{label} SUMMARY")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Recall: {recall:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"FP/image: {fp / len(images):.2f}")


if __name__ == "__main__":
    main()
