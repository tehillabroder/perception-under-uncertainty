"""Audit the DUT test files and freeze a random sample without running a detector."""

import argparse
import json
import math
import random
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

SEED = 20261102
SAMPLE_SIZE = 300
SOURCE = "https://github.com/wangdongdut/DUT-Anti-UAV"


def file_sha256(path):
    return sha256(path.read_bytes()).hexdigest()


def json_sha256(value):
    return sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def required_text(element: ET.Element, path: str) -> str:
    value = element.findtext(path)
    if value is None or not value.strip():
        raise ValueError(f"Missing XML field: {path}")
    return value


def inspect_pair(image_path, xml_path):
    """Validate a pair, including empty annotations, and fingerprint its contents."""
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
    root = ET.parse(xml_path).getroot()
    if root.findtext("filename") != image_path.name:
        raise ValueError("XML filename does not match the image")
    xml_size = (int(required_text(root, "size/width")), int(required_text(root, "size/height")))
    if xml_size != (width, height):
        raise ValueError("XML dimensions do not match the image")
    annotations = []
    edge_boxes = []
    for obj in root.findall("object"):
        box = [
            float(required_text(obj, f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")
        ]
        x1, y1, x2, y2 = box
        if not all(math.isfinite(value) for value in box):
            raise ValueError("non-finite box coordinate")
        if not (0 <= x1 < x2 <= width + 1 and 0 <= y1 < y2 <= height + 1):
            raise ValueError(f"invalid or out-of-bounds box {box}")
        if x2 > width or y2 > height:
            edge_boxes.append(len(annotations))
        annotations.append({"label": required_text(obj, "name"), "box": box})
    return {
        "image": image_path.name,
        "xml": xml_path.name,
        "image_sha256": file_sha256(image_path),
        "xml_sha256": file_sha256(xml_path),
        "rgb_sha256": pixels,
        "image_size": [width, height],
        "annotation_count": len(annotations),
        "annotations": annotations,
        "one_pixel_edge_box_indices": edge_boxes,
        "scene_metadata": {
            key: root.findtext(key)
            for key in ("sequence", "video", "scene", "source/database", "folder")
            if root.findtext(key)
        },
    }


def audit_dataset(images_dir, xml_dir, explored_names):
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    xml_paths = sorted(xml_dir.glob("*.xml"))
    images = {path.stem: path for path in image_paths}
    xmls = {path.stem: path for path in xml_paths}
    errors = []
    if len(images) != len(image_paths):
        errors.append("multiple image files share a stem")
    missing_xml = sorted(set(images) - set(xmls))
    missing_images = sorted(set(xmls) - set(images))
    missing_explored = sorted(set(explored_names) - {path.name for path in image_paths})
    if missing_xml or missing_images or missing_explored:
        errors.append("image/XML coverage or exploratory-image coverage is incomplete")

    records = {}
    for stem in sorted(set(images) & set(xmls)):
        try:
            record = inspect_pair(images[stem], xmls[stem])
            records[record["image"]] = record
        except (OSError, ValueError, TypeError, ET.ParseError) as error:
            errors.append(f"{images[stem].name}: {error}")

    groups = defaultdict(list)
    for name, record in records.items():
        groups[record["rgb_sha256"]].append(name)
    excluded = {}
    eligible = []
    explored = set(explored_names)
    for names in groups.values():
        if explored.intersection(names):
            for name in names:
                excluded[name] = "explored" if name in explored else "exact_copy_of_explored"
        else:
            eligible.append(names[0])
            for name in names[1:]:
                excluded[name] = f"exact_copy_of:{names[0]}"
    eligible.sort()
    metadata = defaultdict(set)
    for record in records.values():
        for key, value in record["scene_metadata"].items():
            metadata[key].add(value)

    audit = {
        "source": SOURCE,
        "image_count": len(image_paths),
        "xml_count": len(xml_paths),
        "validated_pairs": len(records),
        "annotation_count": sum(record["annotation_count"] for record in records.values()),
        "empty_annotation_images": [
            name for name, record in records.items() if record["annotation_count"] == 0
        ],
        "missing_xml": missing_xml,
        "missing_images": missing_images,
        "missing_explored_images": missing_explored,
        "errors": errors,
        "exact_rgb_duplicate_groups": [names for names in groups.values() if len(names) > 1],
        "duplicate_groups_with_different_annotations": [
            names
            for names in groups.values()
            if any(
                records[name]["annotations"] != records[names[0]]["annotations"] for name in names
            )
        ],
        "one_pixel_edge_boxes": {
            name: record["one_pixel_edge_box_indices"]
            for name, record in records.items()
            if record["one_pixel_edge_box_indices"]
        },
        "box_policy": "keep supplied coordinates, including up to one pixel beyond right/bottom",
        "duplicate_policy": "exclude explored RGB groups; keep lowest filename in each other group",
        "metadata_values": {key: sorted(values) for key, values in metadata.items()},
        "independence_limit": (
            "Exact decoded-image copies are excluded. Near-duplicate scenes and training-data "
            "overlap are not established by this audit."
        ),
        "excluded": dict(sorted(excluded.items())),
        "eligible_count": len(eligible),
        "eligible_names": eligible,
        "eligible_names_sha256": json_sha256(eligible),
        "population_sha256": json_sha256(records),
    }
    return audit, records


def make_selection(audit, records, size=SAMPLE_SIZE, seed=SEED):
    if audit["errors"]:
        raise ValueError("Resolve the audit errors before selecting a sample")
    if size < 1 or size > len(audit["eligible_names"]):
        raise ValueError("Sample size must be between 1 and the eligible population size")
    names = sorted(random.Random(seed).sample(audit["eligible_names"], size))
    return names, [records[name] for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/dut_test/img"))
    parser.add_argument("--xml-dir", type=Path, default=Path("data/raw/dut_test/xml"))
    parser.add_argument(
        "--explored-cache", type=Path, default=Path("results/post1/predictions.json")
    )
    parser.add_argument("--original-sample", type=Path, default=Path("data/post1_sample"))
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--expected-images", type=int, default=2200)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-output", type=Path, default=Path("runs/dut_evaluation_audit.json"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/dut_independent_eval.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/post1/evaluation_sample_300.txt")
    )
    args = parser.parse_args()
    if not args.audit_only and (args.manifest.exists() or args.output.exists()):
        parser.error("selection output already exists; do not redraw a frozen sample")
    baseline = json.loads(args.explored_cache.read_text(encoding="utf-8"))
    explored = sorted(baseline["annotations"])
    local_explored = sorted(path.name for path in args.original_sample.glob("*.jpg"))
    if local_explored != explored:
        parser.error("the original sample directory does not match the published exploratory cache")
    audit, records = audit_dataset(args.images_dir, args.xml_dir, explored)
    if audit["image_count"] != args.expected_images:
        audit["errors"].append(f"expected {args.expected_images} test images")
    for name in explored:
        if (
            name in records
            and file_sha256(args.original_sample / name) != records[name]["image_sha256"]
        ):
            audit["errors"].append(f"exploratory image differs from test image: {name}")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"Images={audit['image_count']}, XMLs={audit['xml_count']}, "
        f"UAVs={audit['annotation_count']}, empty images={len(audit['empty_annotation_images'])}, "
        f"eligible={audit['eligible_count']}, errors={len(audit['errors'])}"
    )
    print(f"Audit saved to {args.audit_output}")
    print(
        f"Exact duplicate groups={len(audit['exact_rgb_duplicate_groups'])}, "
        f"differing-annotation groups={len(audit['duplicate_groups_with_different_annotations'])}"
    )
    if audit["errors"]:
        parser.error("audit failed; inspect the saved report before selecting")
    if args.audit_only:
        return
    names, selected_records = make_selection(audit, records, args.size, args.seed)
    list_text = "\n".join(names) + "\n"
    manifest = {
        "schema_version": 1,
        "sample_size": args.size,
        "seed": args.seed,
        "selection_rule": "sorted random.Random(seed).sample(sorted eligible names, size)",
        "source": SOURCE,
        "images_dir": str(args.images_dir),
        "xml_dir": str(args.xml_dir),
        "explored_cache_sha256": file_sha256(args.explored_cache),
        "selection_script_sha256": file_sha256(Path(__file__)),
        "sample_list": str(args.output),
        "sample_list_sha256": sha256(list_text.encode("utf-8")).hexdigest(),
        "audit": audit,
        "images": selected_records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(list_text, encoding="utf-8")
    print(f"Selected {len(names)} images. Saved {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
