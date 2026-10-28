from pathlib import Path
from xml.etree import ElementTree as ET

IMAGES_DIR = Path("data/post1_sample")
XML_DIR = Path("data/raw/dut_test/xml")


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
        if best_index is not None and best_iou >= 0.5:
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


def main():
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {IMAGES_DIR}")

    total_objects = 0
    for image in images:
        xml_path = XML_DIR / f"{image.stem}.xml"
        annotations = read_annotations(xml_path)
        total_objects += len(annotations)
        print(f"{image.name}: {annotations}")
    print(f"\n{len(images)} images, {total_objects} annotated objects")


if __name__ == "__main__":
    main()
