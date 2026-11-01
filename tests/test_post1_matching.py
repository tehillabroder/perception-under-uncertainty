from perception_under_uncertainty.evaluation import match_predictions


def test_duplicate_detection():
    annotations = [{"label": "UAV", "box": [0, 0, 20, 20]}]
    predictions = [
        {"box": [0, 0, 20, 20], "score": 0.9},
        {"box": [0, 0, 20, 20], "score": 0.6},
    ]
    result = match_predictions(predictions, annotations)
    assert result == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "matched_indices": {0},
        "fp_details": [
            {
                "type": "duplicate",
                "score": 0.6,
                "box": [0, 0, 20, 20],
                "best_index": 0,
                "best_iou": 1.0,
            }
        ],
    }


def test_missed_detection():
    annotations = [{"label": "UAV", "box": [0, 0, 20, 20]}]
    predictions = []
    result = match_predictions(predictions, annotations)
    assert result == {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "matched_indices": set(),
        "fp_details": [],
    }
