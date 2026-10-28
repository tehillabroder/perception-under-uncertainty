from scripts.evaluate_post1 import match_predictions


def test_duplicate_detection():
    annotations = [{"label": "UAV", "box": [0, 0, 20, 20]}]
    predictions = [
        {"box": [0, 0, 20, 20], "score": 0.9},
        {"box": [0, 0, 20, 20], "score": 0.6},
    ]
    result = match_predictions(predictions, annotations)
    assert result == {"tp": 1, "fp": 1, "fn": 0}


def test_missed_detection():
    annotations = [{"label": "UAV", "box": [0, 0, 20, 20]}]
    predictions = []
    result = match_predictions(predictions, annotations)
    assert result == {"tp": 0, "fp": 0, "fn": 1}
