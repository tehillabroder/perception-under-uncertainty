# Post 1: Adding full-image prediction to SAHI

**Status: the 30-image exploratory phase is closed.** These findings helped shape the next test; they do not establish what will happen on unseen images.

## What I've tried

I wanted to see whether looking at the whole image as well as smaller slices would help SAHI detect large drones and draw more accurate boxes, particularly in `00967.jpg`.

I used the same 30 images, model weights and settings as before. The only change was enabling full-image prediction with `perform_standard_pred=True`.

I kept the confidence threshold at 0.25 and used the same box-matching rule as before (IoU ≥ 0.5). The images contain 32 annotated drones.

## Results

| Metric | SAHI slices only | SAHI slices + full image |
|---|---:|---:|
| Correct matches | 19 | 19 |
| Incorrect or extra boxes | 39 | 44 |
| Drones without a correct match | 13 | 13 |
| Drones correctly matched (recall) | 59.4% | 59.4% |
| Predictions that were correct (precision) | 32.8% | 30.2% |

Adding the whole image did not increase the total number of correctly matched drones. It produced five more incorrect or extra boxes, so a smaller share of its predictions was correct.

## Checking which drones were matched at confidence 0.25

At confidence 0.25 and IoU 0.5, this change did not improve the overall results on these 30 images.

I then compared which annotated drones were matched in each image, rather than just the totals. There were **no gained or lost drones in any of the 30 images**. Both methods matched exactly the same 19 drones and missed the same 13, including the five large drones missed by slices alone.

## What changed in individual images at confidence 0.25?

Only five images had a change in the number of incorrect or extra boxes:

| Image | Slices only | Slices + full image | Change |
|---|---:|---:|---:|
| `00043.jpg` | 0 | 2 | +2 |
| `00975.jpg` | 2 | 1 | −1 |
| `01475.jpg` | 2 | 3 | +1 |
| `01523.jpg` | 1 | 2 | +1 |
| `01543.jpg` | 3 | 5 | +2 |
| **Total change** | | | **+5** |

The other 25 images had no change in this count. In `00975.jpg`, the extra box counted as a duplicate was no longer present.

SAHI combines predictions from slices and the whole image, so this can change existing boxes rather than simply add new ones. For large drones, the best box overlap with the annotation changed as follows (higher is better):

Here, "large" means the existing 64+ pixel size group: the square root of the annotated box area after scaling the whole image to fit within 640 × 640.

| Image | Slices only | Slices + full image | Correct match? |
|---|---:|---:|---|
| `00301.jpg` | 0.5797 | 0.6781 | Yes in both |
| `00975.jpg` | 0.5682 | 0.7218 | Yes in both |
| `01411.jpg` | 0.2532 | 0.2500 | No in either |
| `01428.jpg` | 0.7313 | 0.6996 | Yes in both |

The boxes in `00605.jpg` and `00967.jpg` were unchanged. In `00605.jpg`, annotation indices 0 and 1 were matched, while index 2 (the third drone) stayed just below the 0.5 matching threshold at 0.4859. In `00967.jpg`, index 0 remained unmatched at 0.4939. These indices start at zero.

So at confidence 0.25, full-image prediction improved some already-correct boxes and removed one duplicate, but recovered none of the missed drones. Overall, it added five net false positives without changing which drones were matched.

## What happened at higher confidence thresholds?

I filtered the saved predictions at the same six confidence thresholds for all four methods. Each cell below shows **recall / total false positives**, with IoU fixed at 0.5.

| Confidence ≥ | YOLO 640 | YOLO 1280 | SAHI slices only | SAHI slices + full image |
|---|---:|---:|---:|---:|
| 0.25 | 68.8% / 9 | 75.0% / 20 | 59.4% / 39 | 59.4% / 44 |
| 0.35 | 68.8% / 4 | 68.8% / 10 | 56.2% / 24 | 56.2% / 26 |
| 0.45 | 65.6% / 3 | 62.5% / 6 | 56.2% / 15 | 56.2% / 15 |
| 0.55 | 62.5% / 1 | 59.4% / 4 | 56.2% / 10 | 56.2% / 9 |
| 0.70 | 53.1% / 0 | 46.9% / 0 | 46.9% / 4 | 50.0% / 4 |
| 0.80 | 34.4% / 0 | 15.6% / 0 | 15.6% / 2 | 25.0% / 4 |

At 0.45 the two SAHI methods had the same counts. At 0.55, full-image prediction kept the same 18 correct matches with one fewer false positive. At 0.70, it retained **16 correct matches rather than 15, with four false positives each**.

This is more nuanced than the first comparison suggested: full-image prediction added five net false positives at 0.25, but retained one additional correct match at the same false-positive count at 0.70. It still did not outperform YOLO 640 on this sample; YOLO 640 at 0.35 kept 22 correct matches with four false positives.

These higher thresholds filter boxes that were already combined and saved at confidence 0.25. I did not rerun SAHI or repeat its box merging at each threshold. Neither this sweep nor the 0.25 comparison tells me what will happen on unseen images.

## Closing this phase

I'm stopping adjustments to SAHI on these 30 deliberately selected images. They were useful for finding problems and forming questions, but repeated checks on the same examples are not an independent evaluation.

The completed 300-image comparison follows this up with a larger, predefined DUT sample and measures processing time as well as detection performance. Both SAHI settings are included in its [write-up](post1_independent_evaluation.md).

## Saved run

The additional run is in `scripts/evaluate_post1_full_image.py`. Predictions for all four methods are saved in `runs/post1_predictions_with_full_image.json`, and the counts use `evaluation.py`. The original results are preserved. This cache and `runs/post1_confidence_sweep_full_image.csv` are explicitly included in git as the frozen follow-up evidence; other run outputs stay local.

The copied cache's `inference_settings.sahi` and `provenance` describe the original baseline. The additional method used the same documented settings with `perform_standard_pred=True`, as recorded in the runner. The original weight hash and environment were not recorded, so exact historical environment and weight identity cannot be verified from that cache.

To repeat the per-image comparison using the saved predictions:

```bash
uv run --locked python -m scripts.analyse_post1 \
  --cache runs/post1_predictions_with_full_image.json
uv run --locked python -m scripts.sweep_post1_confidence \
  --cache runs/post1_predictions_with_full_image.json \
  --output runs/post1_confidence_sweep_full_image_check.csv
```

These commands use saved predictions only. For a deliberate inference rerun, use `uv run --locked python -m scripts.evaluate_post1_full_image`; it needs the original local images and cached weights and writes to `runs/post1_predictions_with_full_image_rerun.json`. An existing output is refused so the frozen evidence is not overwritten.
