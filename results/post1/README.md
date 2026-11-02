# Post1 results

I tried the same drone detector three ways, using yolo at 640, yolo at 1280,
and sahi using smaller image patches. These are the results from 30 hand-picked
images containing 32 drones. It's a small experiment, so the findings are about
this sample.

The 30-image exploratory phase is now closed. A fourth method, SAHI with full-image prediction, was added in a separate cache. It added five net false positives without changing which drones were matched at confidence 0.25, but retained one more correct match at the same false-positive count at 0.70. The [follow-up notes](../../docs/post1_sahi_full_image.md) contain the per-image comparison and four-method sweep.

One interesting result was that raising yolo 640's confidence cutoff from 0.25 to 0.35
kept all 22 correct matches, while false positives dropped from 9 to 4.

## What's here

- [`predictions.json`](predictions.json) contains the saved detections, labelled drone boxes and settings.
- [`confidence_sweep.csv`](confidence_sweep.csv) shows how many drones we keep and false alarms we get as the confidence cutoff rises.
- [`figures/`](figures/) contains the comparison plots. The full set of detection images stays local in `figures/predictions/`.
- [`post1_predictions_with_full_image.json`](../../runs/post1_predictions_with_full_image.json) contains the frozen four-method cache. The original three methods are preserved exactly.
- [`post1_confidence_sweep_full_image.csv`](../../runs/post1_confidence_sweep_full_image.csv) contains results for all four methods at the six predefined thresholds.

The two follow-up files are explicit exceptions to the usual git exclusion for `runs/`. The original cache, sweep and figures remain the three-method baseline.

The model is [TomSmail/drone-yolo-v1](https://huggingface.co/TomSmail/drone-yolo-v1).
The images and labelled boxes come from [DUT Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV).
Some details of the original run weren't recorded, so those fields are left as `null`.
The four-method cache preserves that baseline metadata. Its `inference_settings.sahi` describes sliced-only SAHI. The additional method enables `perform_standard_pred=True`, as documented in the follow-up runner and notes.

## Try it

After `uv sync --locked`, run these commands from the project folder.

```bash
uv run --locked python -m scripts.analyse_post1
uv run --locked python -m scripts.sweep_post1_confidence
```

These commands use the saved predictions, so you don't need the original images or to run the model again. The confidence sweep saves its results to `runs/post1_confidence_sweep.csv`. To analyse a different prediction file, add `--cache` followed by its path, for example `--cache runs/post1_predictions.json`.

A prediction counts as correct when its box overlaps the labelled drone box closely enough, using a fixed overlap score (IoU) of at least 0.5. Each labelled drone can count as a correct match only once.

The sweep checks what happens when we keep only predictions above different confidence levels, from 0.25 to 0.80. Predictions below 0.25 weren't saved, so we can't check lower thresholds without running the model again.

Use these commands for the four-method follow-up.

```bash
uv run --locked python -m scripts.analyse_post1 \
  --cache runs/post1_predictions_with_full_image.json
uv run --locked python -m scripts.sweep_post1_confidence \
  --cache runs/post1_predictions_with_full_image.json \
  --output runs/post1_confidence_sweep_full_image_check.csv
```

The second command writes a fresh comparison table without replacing the frozen sweep. It filters predictions already saved and merged at confidence 0.25. It does not rerun inference or box merging.

## The plots

See the [method comparison](figures/post1_method_comparison.png) and
[false-positive breakdown](figures/post1_error_breakdown.png).

Use these commands to redraw the plots from the saved results.

```bash
uv run --locked python -m scripts.make_post1_visuals
uv run --locked python -m scripts.make_post1_error_breakdown
```

They save into `figures/`. The [evaluation notes](../../docs/post1_xml_eval.md)
walk through the initial findings. The [next experiment plan](../../docs/post1_independent_evaluation.md) covers independent sample selection and comparable processing-time measurements.

[back to the project](../../README.md)
