import pytest

from scripts.evaluate_post1 import box_iou


def test_identical_boxes():
    box = [0, 0, 20, 20]
    assert box_iou(box, box) == 1.0


def test_no_overlap():
    a = [0, 0, 10, 10]
    b = [20, 20, 30, 30]
    assert box_iou(a, b) == 0.0


def test_partial_overlap():
    a = [0, 0, 20, 20]
    b = [10, 10, 30, 30]
    assert box_iou(a, b) == pytest.approx(1 / 7)
