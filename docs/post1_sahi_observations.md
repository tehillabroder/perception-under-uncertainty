# Post 1: SAHI manual observations

This is the historical 30-image visual record. Its manual counts and timings are separate from the automated results in the [300-image write-up](post1_independent_evaluation.md).

## Experimental setup

- Detector: TomSmail `drone-yolo-v1` (pretrained YOLOv8n)
- Dataset: 30 deliberately selected DUT Anti-UAV test images
- Inference: SAHI, 640 × 640 slices, 20% overlap in each direction
- Confidence threshold: 0.25; CPU; sliced-only (`perform_standard_pred=False`)
- Post-processing recorded in `post1_sahi_methodology.md`: GreedyNMM, IOS, 0.5 threshold
- Source of results: manual review of rendered predictions and timings, 28 October 2026

These are **visual judgements**, not one-to-one matches against the dataset XML annotations. Where no failure was noted, I confirmed the predicted drone was correct. A visually correct but oversized bounding box may still fail an IoU-based evaluation.

## Complete 30-image record

| Image | Predictions | Visually correct | Background FPs | Duplicate boxes | Missed drones | CPU time (s) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 00015 | 2 | 1 | 1 | 0 | 0 | 3.42 | FP: small box on cloud |
| 00018 | 1 | 1 | 0 | 0 | 0 | 1.04 | Correct |
| 00028 | 2 | 1 | 1 | 0 | 0 | 1.10 | FP: barbed wire |
| 00039 | 1 | 1 | 0 | 0 | 0 | 1.09 | Correct |
| 00043 | 0 | 0 | 0 | 0 | 1 | 1.08 | Missed drone partly obscured by leaves |
| 00059 | 1 | 1 | 0 | 0 | 0 | 1.05 | Correct |
| 00110 | 1 | 1 | 0 | 0 | 0 | 1.08 | Correct |
| 00152 | 1 | 0 | 1 | 0 | 1 | 1.17 | Difficult drone missed; FP on building |
| 00176 | 1 | 1 | 0 | 0 | 0 | 1.20 | Correct |
| 00301 | 3 | 1 | 1 | 1 | 0 | 1.06 | Double detection on large drone; FP on misty floodlight |
| 00347 | 3 | 1 | 2 | 0 | 0 | 1.08 | FP in sky; FP on drone-like logo |
| 00374 | 3 | 1 | 2 | 0 | 0 | 1.07 | FP on controller and black spot |
| 00605 | 4 | 3 | 1 | 0 | 0 | 0.89 | Three drones detected; bird FP |
| 00667 | 1 | 1 | 0 | 0 | 0 | 1.68 | Correct |
| 00967 | 2 | 1 | 0 | 1 | 0 | 1.03 | Double detection on large drone |
| 00975 | 3 | 1 | 1 | 1 | 0 | 1.18 | Double detection on medium-large drone; car FP |
| 01020 | 1 | 1 | 0 | 0 | 0 | 1.22 | Correct |
| 01023 | 1 | 1 | 0 | 0 | 0 | 1.10 | Correct |
| 01039 | 1 | 1 | 0 | 0 | 0 | 1.19 | Correct |
| 01315 | 2 | 1 | 1 | 0 | 0 | 1.27 | FP on sign |
| 01411 | 2 | 1 | 1 | 0 | 0 | 1.06 | Correct drone but oversized box; FP on pylon |
| 01424 | 1 | 1 | 0 | 0 | 0 | 1.05 | Correct |
| 01428 | 1 | 1 | 0 | 0 | 0 | 1.06 | Correct |
| 01444 | 2 | 1 | 1 | 0 | 0 | 1.52 | FP on tree |
| 01451 | 1 | 1 | 0 | 0 | 0 | 1.41 | Correct |
| 01475 | 2 | 0 | 2 | 0 | 1 | 1.54 | Two small FPs on building; drone missed |
| 01499 | 8 | 1 | 7 | 0 | 0 | 1.64 | Correct drone at confidence 0.71; seven FPs on building and trees |
| 01523 | 1 | 0 | 1 | 0 | 1 | 1.42 | Small red drone missed; large FP on leaves |
| 01535 | 3 | 1 | 2 | 0 | 0 | 1.63 | Correct drone; two leaf FPs |
| 01543 | 3 | 0 | 3 | 0 | 1 | 1.74 | Three FPs on leaves, streetlights and structures; difficult drone missed |
| **Total** | **58** | **27** | **28** | **3** | **5** | **39.07** | **Manual counts only** |

## Preliminary interpretation

- Slicing introduces numerous background false positives in some textured scenes, especially image `01499`.
- Duplicate predictions remain on `00301`, `00967` and `00975` despite slice-level post-processing.
- Difficult or partially obscured targets remain missed in several images.
- A larger input or local crop does not by itself guarantee successful detection. Things like appearance, contrast, occlusion and learned features could be affecting the results.
- These 30 images were chosen for exploration, **not randomly sampled**. They're not an unbiased benchmark.

I've now completed the first XML-based evaluation of these 30 images. The results helped explain why my manual observations and measured recall didn't always agree. Some drones I considered correctly detected had bounding boxes that failed the IoU threshold. I've documented the comparison, localisation errors and remaining questions in the [XML evaluation notes](post1_xml_eval.md).

## References

- Zhao, J., Zhang, J., Li, D. and Wang, D. (2022), *Vision-based Anti-UAV Detection and Tracking*. https://arxiv.org/abs/2205.10851
- DUT Anti-UAV official repository: https://github.com/wangdongdut/DUT-Anti-UAV
- Akyon, F. C., Altinuc, S. O. and Temizel, A. (2022), *Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection*. https://doi.org/10.1109/ICIP46576.2022.9897990
- SAHI documentation: https://obss.github.io/sahi/quick-start/
