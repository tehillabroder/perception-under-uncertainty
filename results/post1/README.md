# Post1 results

I tried the same drone detector three ways: yolo at 640, yolo at 1280,
and sahi using smaller image patches. These are the results from 30 hand-picked
images containing 32 drones. It's a small experiment, so the findings are about
this sample.

One interesting result: raising yolo 640's confidence cutoff from 0.25 to 0.35
kept all 22 correct matches, while false positives dropped from 9 to 4.

## What's here

- [`predictions.json`](predictions.json) — the saved detections, labelled drone boxes and settings.
- [`confidence_sweep.csv`](confidence_sweep.csv) — how many drones we keep and false alarms we get as the confidence cutoff rises.
- [`figures/`](figures/) — the comparison plots. The full set of detection images stays local in `figures/predictions/`.

The model is [TomSmail/drone-yolo-v1](https://huggingface.co/TomSmail/drone-yolo-v1).
The images and labelled boxes come from [DUT Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV).
Some details of the original run weren't recorded; those fields are left as `null`.

## Try it

After `uv sync --locked`, run these from the project folder:

```bash
uv run --locked python -m scripts.analyse_post1
uv run --locked python -m scripts.sweep_post1_confidence
```

They use the saved predictions, so you don't need the images or another model
run. The sweep saves a fresh table to `runs/post1_confidence_sweep.csv`.
To use your own saved run, add `--cache runs/post1_predictions.json`.

The sweep keeps box matching fixed at iou 0.5 — a measure of how well the
predicted box overlaps the labelled one. Each drone can count once.
It tries confidence cutoffs from 0.25 to 0.80. Anything below 0.25 wasn't saved,
so we can't test lower cutoffs with this file.

## The plots

See the [method comparison](figures/post1_method_comparison.png) and
[false-positive breakdown](figures/post1_error_breakdown.png).

To redraw the plots from the saved results:

```bash
uv run --locked python -m scripts.make_post1_visuals
uv run --locked python -m scripts.make_post1_error_breakdown
```

They save into `figures/`. The [evaluation notes](../../docs/post1_xml_eval.md)
walk through what I found and what I'd like to check next.

[back to the project](../../README.md)
