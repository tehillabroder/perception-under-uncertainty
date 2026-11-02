"""
check target identity, preserved baselines and the closed exploratory findings
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from perception_under_uncertainty.evaluation import match_predictions
from scripts.analyse_post1 import analyse

ROOT = Path(__file__).resolve().parents[1]
FULL_IMAGE = "SAHI 640 + full image"


def test_frozen_full_image_cache_preserves_baseline_and_matched_targets():
    baseline = json.loads((ROOT / "results/post1/predictions.json").read_text(encoding="utf-8"))
    combined = json.loads(
        (ROOT / "runs/post1_predictions_with_full_image.json").read_text(encoding="utf-8")
    )
    full_predictions = combined["predictions"].pop(FULL_IMAGE)
    assert combined == baseline
    assert set(full_predictions) == set(baseline["annotations"])
    assert len(full_predictions) == 30

    fp_changes = {}
    gained_at_070 = []
    lost_at_070 = []
    for name, annotations in baseline["annotations"].items():
        sliced = match_predictions(baseline["predictions"]["SAHI 640"][name], annotations)
        full = match_predictions(full_predictions[name], annotations)
        assert full["matched_indices"] == sliced["matched_indices"], name
        if full["fp"] != sliced["fp"]:
            fp_changes[name] = full["fp"] - sliced["fp"]
        sliced_high = match_predictions(
            [p for p in baseline["predictions"]["SAHI 640"][name] if p["score"] >= 0.70],
            annotations,
        )["matched_indices"]
        full_high = match_predictions(
            [p for p in full_predictions[name] if p["score"] >= 0.70], annotations
        )["matched_indices"]
        gained_at_070.extend((name, index) for index in full_high - sliced_high)
        lost_at_070.extend((name, index) for index in sliced_high - full_high)
    assert len(gained_at_070) == 1
    assert not lost_at_070
    assert fp_changes == {
        "00043.jpg": 2,
        "00975.jpg": -1,
        "01475.jpg": 1,
        "01523.jpg": 1,
        "01543.jpg": 2,
    }
    for name in ("00605.jpg", "00967.jpg"):
        assert sorted(p["box"] for p in full_predictions[name]) == sorted(
            p["box"] for p in baseline["predictions"]["SAHI 640"][name]
        )


def test_equal_totals_do_not_hide_gained_and_lost_targets(capsys):
    first = {"box": [0, 0, 100, 100], "score": 0.9}
    second = {"box": [200, 200, 300, 300], "score": 0.9}
    extra = {"box": [400, 400, 500, 500], "score": 0.8}
    data = {
        "model": "example",
        "confidence_threshold": 0.25,
        "annotations": {"example.jpg": [first, second]},
        "image_sizes": {"example.jpg": [640, 640]},
        "predictions": {
            "YOLO 640": {"example.jpg": [first]},
            "YOLO 1280": {"example.jpg": [first]},
            "SAHI 640": {"example.jpg": [first]},
            FULL_IMAGE: {"example.jpg": [second, extra]},
        },
    }
    analyse(data)
    report = capsys.readouterr().out
    assert "example.jpg: gained GT=[1], lost GT=[0], FP=0->1 (+1)" in report
    assert "GT[0] / UAV#1: 100.0px, matched=True->False, best IoU=1.0000->0.0000" in report
    assert "TOTAL: gained=1, lost=1, FP delta=+1" in report


def test_original_three_method_analysis_still_runs(capsys):
    data = json.loads((ROOT / "results/post1/predictions.json").read_text(encoding="utf-8"))
    analyse(data)
    report = capsys.readouterr().out
    assert "TP=19, FP=39, FN=13" in report
    assert FULL_IMAGE not in report


def test_full_image_module_refuses_to_overwrite_existing_output(tmp_path):
    output = tmp_path / "frozen.json"
    output.write_text("preserve this result", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_post1_full_image",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "output already exists" in result.stderr
    assert output.read_text(encoding="utf-8") == "preserve this result"


def test_full_image_runner_changes_only_the_variant_using_cached_weights(tmp_path, monkeypatch):
    from scripts import evaluate_post1_full_image as runner

    baseline = json.loads((ROOT / "results/post1/predictions.json").read_text(encoding="utf-8"))
    original_settings = runner.SAHI_SETTINGS.copy()
    output = tmp_path / "new_run" / "predictions.json"
    model = object()
    calls = []

    def cached_weights(**kwargs):
        assert kwargs == {
            "repo_id": runner.MODEL_REPO,
            "filename": runner.MODEL_FILENAME,
            "local_files_only": True,
        }
        return "cached-best.pt"

    def load_model(**kwargs):
        assert kwargs == {
            "model_type": "ultralytics",
            "model_path": "cached-best.pt",
            "confidence_threshold": 0.25,
            "image_size": 640,
            "device": "cpu",
        }
        return model

    def predict(image_path, detection_model, **kwargs):
        assert detection_model is model
        assert kwargs == {**original_settings, "perform_standard_pred": True, "verbose": 0}
        calls.append(Path(image_path).name)
        return SimpleNamespace(object_prediction_list=[])

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["evaluate_post1_full_image", "--output", str(output)])
    monkeypatch.setattr(runner, "hf_hub_download", cached_weights)
    monkeypatch.setattr(runner.AutoDetectionModel, "from_pretrained", load_model)
    monkeypatch.setattr(runner, "get_sliced_prediction", predict)
    runner.main()

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 30
    assert set(calls) == set(baseline["annotations"])
    assert saved["predictions"].pop(FULL_IMAGE) == dict.fromkeys(calls, [])
    assert saved == baseline
    assert runner.SAHI_SETTINGS == original_settings
