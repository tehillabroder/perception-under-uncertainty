"""match saved detections to annotated boxes without loading a model."""

IOU_THRESHOLD = 0.5

# error groups adapted from tide, section 2.2: https://dbolya.com/tide/paper.pdf
# low overlap is a candidate background error, not a verified cause.
BACKGROUND_IOU_THRESHOLD = 0.1


def box_iou(a, b):
    overlap_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_width * overlap_height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if area_a <= 0 or area_b <= 0:
        raise ValueError("Bounding boxes must have positive area")

    return intersection / (area_a + area_b - intersection)


def prediction_score(prediction):
    return prediction["score"]


def match_predictions(predictions, annotations, iou_threshold=IOU_THRESHOLD):
    """
    match higher-confidence predictions first, using box overlap only.

    each prediction chooses its best-overlapping annotation. if that annotation
    is already claimed, count a duplicate rather than try another annotation.
    """
    predictions = sorted(predictions, key=prediction_score, reverse=True)
    matched_annotations = set()
    true_positives = 0
    false_positives = 0
    fp_details = []

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

            if best_index is not None and best_iou >= iou_threshold:
                error_type = "duplicate"
            elif best_iou >= BACKGROUND_IOU_THRESHOLD:
                error_type = "localisation_candidate"
            else:
                # fragments of large drones can also have very little overlap.
                error_type = "background_candidate"

            fp_details.append(
                {
                    "type": error_type,
                    "score": prediction["score"],
                    "box": prediction["box"],
                    "best_index": best_index,
                    "best_iou": best_iou,
                }
            )

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": len(annotations) - len(matched_annotations),
        "matched_indices": matched_annotations,
        "fp_details": fp_details,
    }
