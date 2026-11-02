# Post 1: Next Experiment — An Independent DUT Sample

## Why I'm moving on

The [30-image exploration](post1_sahi_full_image.md) is closed. Full-image prediction added five net false positives without recovering a drone at confidence 0.25. At 0.70, it retained one additional correct match at the same false-positive count. Neither result establishes how either method will perform on new images.

The next question is whether those patterns hold on a larger sample chosen before looking at its predictions, and how much processing time each method needs.

## Next chunk: select and freeze the sample

**Status: planned; the independent sample has not been selected or evaluated.**

The proposed starting point is 300 images from the official DUT detection test split, selected without replacement using seed `20261102`. I will confirm the available pool and freeze the selection before running the detector.

1. Record the dataset release/source and inventory the test images and XML annotations. Check that files are readable and boxes are valid. Record exclusions and reasons; do not silently replace failed inputs during evaluation.
2. Exclude all 30 filenames in `results/post1/predictions.json`, any other images already used to choose settings, and exact copies of those images under different names.
3. Check the release's sequence/source information and duplicate relationships. Where related frames can be identified, keep exploratory and evaluation groups separate. Different filenames alone do not establish independence; record what grouping information is available and any remaining uncertainty.
4. Select from the sorted eligible pool using the fixed seed, without choosing images by appearance or detector output. If group-based selection is needed, fix that rule before drawing the sample. If fewer than 300 eligible images remain, record the shortfall and settle the sample size before inference.
5. Save a versioned manifest in `configs/dut_independent_eval.json`: source, selection rule, seed, eligible-pool checksum, exclusions, selected filenames, and image/XML checksums. Commit it before the evaluation run.
6. Test that the selection repeats exactly, has no duplicate IDs or excluded images, and has a valid annotation file for every image. Keep valid images with no drones if the source split contains them.

This is independence from our exploratory choices. I also need to check the pretrained model's documented training data before describing these images as unseen by the model itself.

## Keep the comparison fixed

- Use the same model weights for all four methods: YOLO 640, YOLO 1280, SAHI 640, and SAHI 640 + full image. Record the resolved model revision and weight checksum.
- Keep the existing slice size, overlap and box-merging settings. The two SAHI settings differ only in `perform_standard_pred`.
- Save predictions at confidence 0.25, then report the already-chosen thresholds: 0.25, 0.35, 0.45, 0.55, 0.70 and 0.80. This is filtering saved, merged boxes, not rerunning merging at each threshold.
- Keep the primary matching threshold at IoU 0.5. Reuse `evaluation.py`: predictions are considered in confidence order; each chooses its best-overlapping annotation; a previously claimed annotation counts as a duplicate. Matching checks box overlap, not drone subtype labels.
- Report correct matches, false positives, missed drones, recall, precision and false positives per image, plus the existing size groups and matched-target gains/losses. Compare recall at similar false-positive counts, not only at equal confidence thresholds.
- Report all predefined thresholds. Do not choose new settings from these evaluation results; further tuning would need separate development data and another untouched evaluation sample.

## Measure processing time consistently

Use one recorded device and software environment for every method, with batch size one. Load the model once and perform five untimed warm-up calls per method on images outside the evaluation sample.

For the main timing, decode each image before starting the clock, then measure from the loaded image to final boxes. Include resizing, slicing, all model calls and box merging. Exclude model download/loading, image decoding, XML matching, plotting and saving files. Synchronise the device around timing if using an accelerator.

Run three timed passes per method, retaining per-image timings and method order. Save predictions from the first pass. Report median and 95th-percentile milliseconds per image, with the timing boundary and hardware alongside them. The old manual timings are not a comparable latency benchmark.

## Before the fuller run

The current inference scripts are specific to the 30-image exploration. The next implementation needs to consume the frozen manifest, record the agreed timings and complete run settings, and write to a separate experiment directory. A small smoke test should check sample coverage, all four methods, timing records and unchanged source files before the full run.

The next deliverable is the checked, committed sample manifest and selection procedure. The larger inference run follows that step.
