# Post 1: SAHI experimental methodology

The original setup and exploratory notes are preserved below. The [completed 300-image evaluation](#the-300-image-evaluation) has its own section, including the final matching and timing procedure.

## Method

I am investigating sliced inference using SAHI (Akyon et al., 2022) with the pretrained TomSmail YOLOv8n detector on DUT Anti-UAV.

Initial configuration:
- Slice dimensions: 640 × 640 pixels
- Overlap: 20% horizontally and vertically
- Model input size: 640
- Confidence threshold: 0.25
- Device: CPU
- Full-image prediction within SAHI: disabled
- Post-processing: GreedyNMM, IOS, threshold 0.5

## Parameter justification

640 × 640 was selected to match the detector's documented training input resolution and the slice dimensions used in the original SAHI paper's VisDrone experiments.

20% overlap follows the documented SAHI defaults. The original paper evaluated 25% overlap.

Full-image prediction is not used initially so I can isolate sliced inference from the existing whole-image baselines.

## Research references

1. Akyon, F. C., Altinuc, S. O., and Temizel, A. (2022). *Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection*. IEEE ICIP. https://doi.org/10.1109/ICIP46576.2022.9897990

2. Ultralytics. *SAHI Tiled Inference Guide*. https://docs.ultralytics.com/guides/sahi-tiled-inference

3. SAHI developers. *How Sliced Inference Works*. https://obss.github.io/sahi/guides/sliced-inference/

4. SAHI API Reference. https://obss.github.io/sahi/api/

5. TomSmail. *drone-yolo-v1*, model card and pretrained weights. https://huggingface.co/TomSmail/drone-yolo-v1

Documentation consulted: 28 October 2026.

## Original exploratory findings

I have some evidence that retaining more image detail does not NECESSARILY produce better detections. SAHI changes both the effective object scale and the surrounding context. Those effects are not easily separated. A model may see a small object more clearly, but also mistake leaves, branches and other structures for drones.

Looking at three 'interesting' cases in a bit more detail:

00043: SAHI still cannot detect the partially obscured drone. No predictions at all.

01499: SAHI recovered the drone, but generated seven false positives. YOLO at 1280 had already recovered it with only one false positive. Slicing may be removing useful background context.

01523: The single predicted quadcopter box is enormous, spanning leaves/trees and part of the building. Might be bad localisation from the detector or an issue with how predictions are processed.

Comparison after checking on the 30 selected images:

| Metric | YOLO 640 | YOLO 1280 | SAHI 640 |
|---|---|---|---|
| Missed drones | 6 | 4 | 5 |
| Incorrect/extra boxes | 5 | 12 | 31 |

These are my manual counts, i did not formally match them against the XML yet.

- SAHI has generated over six times as many incorrect or duplicate boxes as whole-image inference at 640.
- 7 of SAHI's background false positives came from 01499 alone. Perhaps a small number of cluttered scenes dominate the errors.
- the duplicates on 00301, 00967 and 00975 make me want to check SAHI's post-processing, not just the classifications.
- Difficult to spot and obscured drones are still missed. Like i noted earlier, resolution may not be the main limiting factor.

Now id like to
1. implement an automated evaluator using the DUT XML annotations, with one-to-one bounding-box matching at IoU 0.5, recording recall by object size, false positives and duplicates.
2. record comparable inference timings for the 640 and 1280 baselines.

At this point, manual inspection has shown some interesting failure modes, but now i need to establish some reproducible measurements to clarify whether those patterns hold.

The emerging question is whether slicing improves small-object observability enough to outweigh the additional background confusion and computational cost.

## The 300-image evaluation

The sections above record the original exploration, including provisional manual counts. The later [300-image write-up](post1_independent_evaluation.md) uses automated matching against the dataset annotations.

### Selection and settings

I selected 300 unique images containing 309 annotated drones using seed `20261102`, before running predictions. The audit of 2,200 DUT test images found 539 exact duplicate pairs. Removing duplicate copies and excluding the original exploration left 1,631 eligible images. Nine annotation files in the audited test set had boxes extending one pixel beyond an edge; I kept those annotations unchanged.

The [selection record](../configs/dut_independent_eval.json) and [image list](../results/post1/evaluation_sample_300.txt) preserve the inputs. All four methods use the same TomSmail weights. YOLO uses whole-image sizes of 640 and 1280; SAHI uses 640 × 640 slices with 20% overlap and a detector input size of 640. Predictions are combined with GreedyNMM, using intersection over smaller area (IOS) at 0.5. The second SAHI variant enables `perform_standard_pred=True` to include a whole-image prediction at 640.

The starting confidence cutoff is 0.25. The sweep filters saved boxes at 0.35, 0.45, 0.55, 0.70 and 0.80, without rerunning prediction or box combining. The main match threshold is IoU 0.5. The [evaluator](../src/perception_under_uncertainty/evaluation.py) processes predictions by descending confidence, choosing each box's best-overlapping annotation. It requires IoU ≥ 0.5 for a match; a later box meeting that threshold for an already matched annotation counts as a duplicate. It does not try a second annotation instead. Matching uses box overlap, treating the detector's predictions as UAV candidates rather than comparing class names.

### Timing and reproducibility

All methods ran on the same Intel Core i5-1038NG7 CPU, one image at a time. Each had five untimed warm-up calls followed by three timed passes. Method order rotated between passes; image order was shuffled with the sample seed plus the pass index and shared by all methods within each pass.

Timing starts with a loaded RGB image and ends with serialisable final predictions. It includes preparation, slicing, model calls and box combining, and excludes image reading/decoding, model loading, matching and file writes.

I took each image's median across the three passes, then calculated the median and 90th percentile across the 300 per-image values. The 90th percentile is the time at or below which 90% of these values fall, using linear interpolation.

| Method | Median time per image | 90th percentile |
|---|---:|---:|
| YOLO 640 | 94 ms | 97 ms |
| YOLO 1280 | 262 ms | 271 ms |
| SAHI 640 | 1,226 ms | 1,256 ms |
| SAHI 640 + full image | 1,338 ms | 1,398 ms |

Source: [summary.csv](../runs/dut_300_evaluation/summary.csv), calculated from the 3,600 entries in [latencies.csv](../runs/dut_300_evaluation/latencies.csv). Predictions are retained from the first pass; the other passes supply timings.

The [run manifest](../runs/dut_300_evaluation/run_manifest.json) records the completed status, source and model hashes, CPU, package versions and timing order. It contains the same complete record as [predictions.json](../runs/dut_300_evaluation/predictions.json), because the runner writes the finished cache to both files. The [results guide](../runs/dut_300_evaluation/README.md) contains the commands for repeating the saved-prediction analysis.
