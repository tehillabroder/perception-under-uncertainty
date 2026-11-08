# The 300-image results behind Post 1

Start with [the write-up: more pixels, more drones?](../../docs/post1_independent_evaluation.md). This folder keeps the evidence behind it: four methods, 300 images and 309 labelled drones from DUT Anti-UAV.

## The five run files

| File | What's in it |
|---|---|
| [summary.csv](summary.csv) | Matches, misses, false positives, recall, precision and CPU timings at confidence 0.25 and IoU 0.5. |
| [confidence_sweep.csv](confidence_sweep.csv) | All four methods at the six predefined confidence cutoffs, from 0.25 to 0.80. |
| [predictions.json](predictions.json) | The saved boxes and confidence scores, annotations, image sizes and run details. Enough to repeat the detection analysis without the photographs or model. |
| [latencies.csv](latencies.csv) | All 3,600 timings: 300 images × four methods × three passes. |
| [run_manifest.json](run_manifest.json) | The completed run record, including model/source hashes, CPU, package versions and timing order. The runner saved the same complete record here and in `predictions.json`. |

These are the original outputs from the completed evaluation on 2 November 2026. The [image list](../../results/post1/evaluation_sample_300.txt), [selection record](../../configs/dut_independent_eval.json) and [runner](../../scripts/evaluate_post1_sample.py) were committed before the evaluation. The visualisations were made afterwards from these saved predictions.

## Evaluation slide

[One comparison slide](visuals/post1_evaluation_comparison.png) puts two aligned charts side by side:

- **Which method found more drones?** Share of the 309 labelled drones correctly detected, on a 0–100% scale.
- **Which method produced more extra boxes?** Extra detections per image, including inaccurate and duplicate boxes. This is a count per image, not a percentage.

The slide compares all four methods at confidence 0.25 and matching IoU 0.50, using shared method rows and colours. Its takeaway: YOLO 1280 found the most drones. Recreate it from saved predictions, without source images or inference:

```bash
uv run --locked python -m scripts.make_post1_evaluation_visuals
```

Use `--cache` for another prediction cache and `--output-dir` to save charts elsewhere; the default output is `visuals/` beside the cache.

## Three cases to start with

- [00395: where slicing helped](visuals/00395_comparison.png). SAHI passes the overlap rule; YOLO 1280 is just below it.
- [00224: a box around the drone isn't necessarily a good fit](visuals/00224_comparison.png). Both SAHI variants extend well beyond the labelled box.
- [00150: adding the whole image helps](visuals/00150_comparison.png). One final SAHI box becomes a match, although an extra box remains.

Supporting examples: [00622, a match lost after adding full-image prediction](visuals/00622_comparison.png), and [00787, background detections on tree branches](visuals/00787_comparison.png).

The first four figures use target-centred crops. The last shows the full scene. Counts and heading scores refer to all predictions in the image, including any outside a crop. Dashed boxes are dataset annotations; solid boxes are predictions. Images and annotations are from [DUT Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV), Zhao et al. (2022); the detector is [TomSmail/drone-yolo-v1](https://huggingface.co/TomSmail/drone-yolo-v1).

## Repeat the analysis

From the repository root, after `uv sync --locked`:

```bash
uv run --locked python -m scripts.analyse_post1 \
  --cache runs/dut_300_evaluation/predictions.json

uv run --locked python -m scripts.sweep_post1_confidence \
  --cache runs/dut_300_evaluation/predictions.json \
  --output runs/dut_300_confidence_check.csv
```

The first command prints missed drones, gained/lost targets, size-group recall, error groups and the IoU sensitivity comparison. The second recreates the confidence table in a separate file. Both use saved boxes; neither runs inference or repeats box combining.

To redraw the five figures, the original JPGs are needed locally:

```bash
uv run --locked python -m scripts.visualise_post1_cases
```

The default image folder is `data/raw/dut_test/img`. Use `--images-dir` for another location and `--output-dir` to save renders elsewhere.

The timings cover loaded-image preparation through to final boxes, including slicing and combining. They exclude image decoding, model loading and file writes. The summary uses the median of three timings for each image, then the median and 90th percentile across those 300 values.

For a deliberate inference rerun, use a fresh output folder:

```bash
uv run --locked python -m scripts.evaluate_post1_sample \
  --output-dir runs/dut_300_evaluation_rerun
```

That command needs the local DUT images, XML annotations and cached model weights. The existing run stays in this folder.
