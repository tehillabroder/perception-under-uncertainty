# Post 1. A New DUT Evaluation Sample

## What I want to find out

The [30-image exploration](post1_sahi_full_image.md) is closed. Adding full-image prediction to SAHI produced more false positives at confidence 0.25, but retained one more correct match at the same false-positive count at 0.70.

I now want to see whether these patterns hold on a larger sample. Do larger inputs and image slicing find more drones accurately, and how much extra processing time do they need?

## The sample is fixed

I randomly selected **300 unique images containing 309 labelled drones** from the DUT test set. I used seed `20261102` so the selection can be repeated, and chose the images before running any predictions on them.

Checking the 2,200 test images revealed **539 exact duplicate pairs**, including copies of five images from the original exploration. After removing duplicate copies and excluding the original 30 images, there were 1,631 unique images to choose from.

Every image had a matching annotation file. Nine files had boxes extending one pixel beyond an image edge. I kept those annotations unchanged and recorded the issue.

The [image list](../results/post1/evaluation_sample_300.txt) and [full selection record](../configs/dut_independent_eval.json) are saved for the commit. The 300-image evaluation has not been run. A small software check has passed using two original images instead.

## What this sample can and cannot tell us

All images in this test set contain drones. The false-positive counts will tell us about extra or incorrect detections in those images, not how often the model raises an alarm when no drone is present.

Exact copies are excluded, but similar scenes may remain because the dataset does not identify which images belong to the same scene. I also cannot confirm whether any images appeared in the model's training data. This is a fresh sample for our experiment, not a guarantee of completely unseen scenes.

## Fixed comparison

I will compare **YOLO 640, YOLO 1280, SAHI 640 and SAHI 640 + full image**, using the same model weights and settings as before.

The main comparison stays at confidence 0.25 and a box-overlap score of at least 0.5 (IoU). Each labelled drone can count as a correct match only once. I will report correct matches, missed drones, false positives, recall and precision, including how results differ by drone size.

I will also filter the saved predictions at confidence levels 0.35, 0.45, 0.55, 0.70 and 0.80. This will show whether one method keeps more correct matches at a similar false-positive count. It does not rerun the model or combine the boxes again.

The settings are fixed before the evaluation. I will report all these thresholds rather than use the new sample to keep adjusting SAHI.

## Processing time

All four methods will run on the same Intel Core i5-1038NG7 CPU, one image at a time. After five untimed warm-up calls per method, each image will be processed three times. The methods use the same image order within each pass.

Timing starts once the image is loaded and ends when the final boxes are ready. It includes image preparation, slicing, prediction and box merging, but excludes loading the model, reading the image from disk and saving results.

For each image, I will take the middle of its three timings. Across those 300 values, I will report the **median**, or typical time, and the **90th percentile**, the time at or below which 90% of those values fall. These measurements describe this CPU setup, not performance on other hardware.

## Commands

Run these from the repository root after `uv sync --locked`.

Check all 300 selected inputs without running a detector.

```bash
uv run --locked python -m scripts.evaluate_post1_sample --validate-only
```

After committing the selection and code, run the full comparison.

```bash
uv run --locked python -m scripts.evaluate_post1_sample \
  --output-dir runs/dut_300_evaluation
```

Use a new output directory. It will contain the predictions, individual timings, main results and confidence comparisons, along with a record of the run settings.

Analyse the resulting predictions with the existing script.

```bash
uv run --locked python -m scripts.analyse_post1 \
  --cache runs/dut_300_evaluation/predictions.json
```
