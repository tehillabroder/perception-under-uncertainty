"""Check unbiased eligibility, duplicate exclusions and reproducible sample selection."""

import json
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from scripts.select_post1_evaluation import audit_dataset, inspect_pair, make_selection


def write_pair(root, name, color, boxes=()):
    images = root / "img"
    xmls = root / "xml"
    images.mkdir(exist_ok=True)
    xmls.mkdir(exist_ok=True)
    Image.new("RGB", (20, 10), color).save(images / f"{name}.jpg")
    objects = "".join(
        "<object><name>UAV</name><bndbox>"
        + "".join(
            f"<{key}>{value}</{key}>"
            for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box, strict=True)
        )
        + "</bndbox></object>"
        for box in boxes
    )
    (xmls / f"{name}.xml").write_text(
        f"<annotation><filename>{name}.jpg</filename>"
        f"<size><width>20</width><height>10</height></size>{objects}</annotation>",
        encoding="utf-8",
    )
    return images, xmls


def test_empty_images_are_eligible_and_all_exploratory_copies_are_excluded(tmp_path):
    write_pair(tmp_path, "explored", "red", [(1, 1, 10, 5)])
    write_pair(tmp_path, "copy", "red", [(1, 1, 10, 5)])
    write_pair(tmp_path, "negative", "blue")
    write_pair(tmp_path, "target", "green", [(1, 1, 10, 5)])
    images, xmls = write_pair(tmp_path, "target_copy", "green", [(1, 1, 10, 5)])
    audit, records = audit_dataset(images, xmls, ["explored.jpg"])
    assert not audit["errors"]
    assert "negative.jpg" in audit["eligible_names"]
    assert audit["eligible_names"] == ["negative.jpg", "target.jpg"]
    assert audit["excluded"]["copy.jpg"] == "exact_copy_of_explored"
    assert make_selection(audit, records, 2, 42) == make_selection(audit, records, 2, 42)
    with pytest.raises(ValueError, match="Sample size"):
        make_selection(audit, records, 3)


def test_missing_xml_is_not_silently_treated_as_a_negative_or_dropped(tmp_path):
    images, xmls = write_pair(tmp_path, "unlabelled", "blue")
    (xmls / "unlabelled.xml").unlink()
    audit, records = audit_dataset(images, xmls, [])
    assert audit["missing_xml"] == ["unlabelled"]
    with pytest.raises(ValueError, match="audit errors"):
        make_selection(audit, records, 1)


def test_one_pixel_edge_boxes_are_recorded_without_changing_coordinates(tmp_path):
    images, xmls = write_pair(tmp_path, "edge", "blue", [(1, 1, 21, 11)])
    record = inspect_pair(images / "edge.jpg", xmls / "edge.xml")
    assert record["annotations"][0]["box"] == [1, 1, 21, 11]
    assert record["one_pixel_edge_box_indices"] == [0]
    write_pair(tmp_path, "edge", "blue", [(1, 1, 22, 11)])
    with pytest.raises(ValueError, match="out-of-bounds"):
        inspect_pair(images / "edge.jpg", xmls / "edge.xml")


@pytest.mark.parametrize(
    "field", ["size/width", "size/height", "object/bndbox/xmin", "object/name"]
)
def test_missing_xml_fields_are_reported_as_audit_errors(tmp_path, field):
    images, xmls = write_pair(tmp_path, "incomplete", "blue", [(1, 1, 10, 5)])
    path = xmls / "incomplete.xml"
    tree = ET.parse(path)
    element = tree.find(field)
    assert element is not None
    element.text = None
    tree.write(path, encoding="utf-8")
    audit, records = audit_dataset(images, xmls, [])
    assert not records
    assert "Missing XML field" in audit["errors"][0]
    with pytest.raises(ValueError, match="audit errors"):
        make_selection(audit, records, 1)


def test_frozen_sample_matches_selection_rule_and_has_no_exploratory_content():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/dut_independent_eval.json").read_text(encoding="utf-8"))
    names = [record["image"] for record in manifest["images"]]
    expected = sorted(
        random.Random(manifest["seed"]).sample(
            manifest["audit"]["eligible_names"], manifest["sample_size"]
        )
    )
    assert names == expected
    assert len(names) == len(set(names)) == 300
    assert len({record["rgb_sha256"] for record in manifest["images"]}) == 300
    assert not set(names).intersection(manifest["audit"]["excluded"])
    assert (root / manifest["sample_list"]).read_text(encoding="utf-8").splitlines() == names
    assert manifest["audit"]["image_count"] == manifest["audit"]["xml_count"] == 2200
    assert not manifest["audit"]["errors"]
