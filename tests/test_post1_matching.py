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


def test_highest_score_claims_best_target_without_falling_back_to_another():
    annotations = [{"box": [0, 0, 20, 20]}, {"box": [10, 0, 30, 20]}]
    predictions = [
        {"box": [5, 0, 25, 20], "score": 0.5},
        {"box": [0, 0, 20, 20], "score": 0.9},
    ]
    result = match_predictions(predictions, annotations)
    assert result["matched_indices"] == {0}
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["fp_details"][0]["type"] == "duplicate"
    assert result["fp_details"][0]["score"] == 0.5


def test_iou_boundary_matches_without_requiring_subtype_label_agreement():
    result = match_predictions(
        [{"box": [0, 0, 10, 20], "score": 0.9, "label": "quadcopter"}],
        [{"box": [0, 0, 20, 20], "label": "UAV"}],
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
