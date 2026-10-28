# Post 1: SAHI Experimental Methodology

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

## Findings as I work

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