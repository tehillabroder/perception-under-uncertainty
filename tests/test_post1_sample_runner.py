"""Check sample integrity, four-method timing boundaries and latency summaries."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import evaluate_post1, evaluate_post1_sample
from scripts.select_post1_evaluation import file_sha256, inspect_pair


@pytest.fixture
def sample(tmp_path):
    images = tmp_path / "img"
    xmls = tmp_path / "xml"
    images.mkdir()
    xmls.mkdir()
    image = images / "a.jpg"
    xml = xmls / "a.xml"
    Image.new("RGB", (20, 10), "blue").save(image)
    xml.write_text(
        "<annotation><filename>a.jpg</filename><size><width>20</width>"
        "<height>10</height></size></annotation>",
        encoding="utf-8",
    )
    sample_list = tmp_path / "sample.txt"
    sample_list.write_text("a.jpg\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "sample_size": 1,
        "sample_list": str(sample_list),
        "sample_list_sha256": file_sha256(sample_list),
        "audit": {"excluded": {}},
        "images": [inspect_pair(image, xml)],
    }
    return manifest, images, xmls


@pytest.mark.parametrize("changed", ["image", "xml", "list"])
def test_changed_inputs_are_rejected_before_inference(sample, changed):
    manifest, images, xmls = sample
    assert evaluate_post1_sample.validate_sample(manifest, images, xmls) == manifest["images"]
    if changed == "image":
        Image.new("RGB", (20, 10), "red").save(images / "a.jpg")
    elif changed == "xml":
        path = xmls / "a.xml"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        Path(manifest["sample_list"]).write_text("another.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch|differs"):
        evaluate_post1_sample.validate_sample(manifest, images, xmls)


def test_timing_excludes_loading_and_warmup_and_retains_only_first_pass(monkeypatch):
    now = [0.0]
    calls = Counter()

    def load(path):
        now[0] += 100
        return path.name

    def make_predictor(method):
        def predict(image):
            calls[method] += 1
            now[0] += 0.001
            return [{"call": calls[method], "image": image}]

        return predict

    monkeypatch.setattr(evaluate_post1_sample, "load_rgb", load)
    monkeypatch.setattr(evaluate_post1_sample, "perf_counter", lambda: now[0])
    predictors = {method: make_predictor(method) for method in evaluate_post1_sample.METHODS}
    records = [{"image": "a.jpg"}, {"image": "b.jpg"}]
    predictions, timings, orders = evaluate_post1_sample.collect_predictions(
        records, Path("images"), predictors, [Path("warm.jpg")], passes=3
    )
    assert len(timings) == 4 * 2 * 3
    assert all(row["seconds"] == pytest.approx(0.001) for row in timings)
    assert set(calls.values()) == {7}
    for method in predictors:
        assert set(predictions[method]) == {"a.jpg", "b.jpg"}
        assert sorted(value[0]["call"] for value in predictions[method].values()) == [2, 3]
        for order in orders:
            actual_order = [
                row["image"]
                for row in timings
                if row["method"] == method and row["pass"] == order["pass"]
            ]
            assert actual_order == order["images"]
    assert orders[1]["methods"] == list(predictors)[1:] + list(predictors)[:1]
    with pytest.raises(ValueError, match="Warm-up"):
        evaluate_post1_sample.collect_predictions(
            records, Path("images"), predictors, [Path("a.jpg")]
        )


def test_latency_percentiles_use_per_image_medians_and_keep_negative_images():
    detection = {"box": [0, 0, 10, 10], "score": 0.9}
    cache = {
        "annotations": {"a.jpg": [detection], "b.jpg": []},
        "predictions": {"example": {"a.jpg": [detection], "b.jpg": [detection]}},
    }
    timings = [
        {"method": "example", "image": name, "seconds": seconds}
        for name, values in (("a.jpg", (0.001, 0.002, 0.1)), ("b.jpg", (0.010, 0.012, 0.1)))
        for seconds in values
    ]
    row = evaluate_post1_sample.summarise(cache, timings)[0]
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 0)
    assert row["precision"] == row["fp_per_image"] == 0.5
    assert row["median_ms"] == pytest.approx(7)
    assert row["p90_ms"] == pytest.approx(11)


def test_loaded_rgb_inputs_and_local_sahi_override_preserve_baseline_defaults(monkeypatch):
    image = Image.new("RGB", (20, 10), "blue")
    sources = []

    def yolo_predict(**kwargs):
        sources.append(kwargs["source"])
        assert kwargs["device"] == "cpu"
        assert kwargs["conf"] == 0.25
        return [SimpleNamespace(boxes=None)]

    yolo = SimpleNamespace(predict=yolo_predict)
    evaluate_post1.predict_yolo(yolo, image, 640)
    evaluate_post1.predict_yolo(yolo, Path("a.jpg"), 640)
    assert sources == [image, "a.jpg"]
    settings = evaluate_post1.SAHI_SETTINGS.copy()

    def sliced(source, model, **kwargs):
        sources.append(source)
        assert kwargs == {**settings, "perform_standard_pred": True, "verbose": 0}
        return SimpleNamespace(object_prediction_list=[])

    monkeypatch.setattr(evaluate_post1, "get_sliced_prediction", sliced)
    evaluate_post1.predict_sahi(None, image, settings={**settings, "perform_standard_pred": True})
    assert sources[-1] is image
    assert evaluate_post1.SAHI_SETTINGS == settings
