"""check filtering boundaries and the published measurements."""

import csv
import json
from pathlib import Path

import pytest

from scripts.sweep_post1_confidence import confidence_sweep

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "post1"
RUNS_DIR = RESULTS_DIR.parents[1] / "runs"


def test_filtering_keeps_boundary_scores_and_counts_empty_images():
    data = {
        "confidence_threshold": 0.25,
        "annotations": {"target.jpg": [{"box": [0, 0, 20, 20]}], "empty.jpg": []},
        "predictions": {
            "example": {
                "target.jpg": [{"box": [0, 0, 20, 20], "score": 0.5}],
                "empty.jpg": [{"box": [0, 0, 20, 20], "score": 0.3}],
            }
        },
    }
    rows = confidence_sweep(data, (0.25, 0.5, 0.8))
    assert [(row["tp"], row["fp"], row["fn"]) for row in rows] == [(1, 1, 0), (1, 0, 0), (0, 0, 1)]
    assert rows[0]["fp_per_image"] == 0.5
    assert rows[1]["recall"] == 1.0
    assert rows[2]["recall"] == 0.0
    with pytest.raises(ValueError, match="saved confidence cutoff"):
        confidence_sweep(data, (0.2,))


@pytest.mark.parametrize(
    ("cache_path", "sweep_path", "expected_rows"),
    [
        (RESULTS_DIR / "predictions.json", RESULTS_DIR / "confidence_sweep.csv", 18),
        (
            RUNS_DIR / "post1_predictions_with_full_image.json",
            RUNS_DIR / "post1_confidence_sweep_full_image.csv",
            24,
        ),
    ],
)
def test_published_sweep_matches_saved_predictions(cache_path, sweep_path, expected_rows):
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    actual = confidence_sweep(data)
    with sweep_path.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert len(actual) == len(saved) == expected_rows
    for row, expected in zip(actual, saved, strict=True):
        assert row["method"] == expected["method"]
        for key in row.keys() - {"method"}:
            assert row[key] == pytest.approx(float(expected[key]))

    baseline = [row for row in actual if row["confidence"] == 0.25]
    assert [(row["tp"], row["fp"], row["fn"]) for row in baseline[:3]] == [
        (22, 9, 10),
        (24, 20, 8),
        (19, 39, 13),
    ]
    if expected_rows == 24:
        full_image = [row for row in actual if row["method"] == "SAHI 640 + full image"]
        assert [(row["tp"], row["fp"], row["fn"]) for row in full_image] == [
            (19, 44, 13),
            (18, 26, 14),
            (18, 15, 14),
            (18, 9, 14),
            (16, 4, 16),
            (8, 4, 24),
        ]
