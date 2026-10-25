# experiment plan

i'm planning six experiments, shared through linkedin posts over
a few weeks. each one will have a question, a baseline comparison and a plot or
short silent clip showing what happened.

## 1. small objects and observability

what happens to tiny objects when an image is resized?

i'll compare a 640 by 640 whole-image detector with sahi sliced inference on a
subset of dut anti-uav. i'll use the same model and weights, group objects by
pixel size, and measure recall, false positives and processing time. the first
milestone is this comparison running end to end with one diagnostic plot.

## 2. temporal cues and clutter

when does motion make an object easier to find?

i'll try three-frame differencing and dense optical flow on static-camera and
moving-camera sequences. i'm interested in how parallax and jitter affect false
alarms. i'll keep frame timing and any use of future frames explicit.

## 3. monocular geometry

what can a single camera tell us about an object's direction and distance?

i'll use camera intrinsics to calculate bearings and angular rates, then measure
their sensitivity to detection jitter and camera motion. for size-based depth,
i'll vary the box dimensions and show how the assumed object size affects the
estimate. the plot will state the calibration, units and assumptions used.

## 4. rgb and thermal fusion

does combining two sensors help when one image becomes harder to interpret?

i'll inspect the paired streams in anti-uav300, establish their coordinate and
timing relationship, and compare each sensor with confidence-weighted late fusion.
controlled glare, noise, blur and alignment offsets will let me examine where the
combined result improves or gets worse.

## 5. detection calibration and abstention

how well does a detector's confidence describe whether its predictions are useful?

this is the centre of the project. i'll extend my classification-calibration work
to detections, defining correctness through class agreement and box overlap.
localisation uncertainty will get its own evaluation. calibration will use held-out
sequences, followed by clean and degraded test data. i'll compare error against
coverage as an abstention threshold changes.

## 6. uncertainty and tracking

how do detection errors affect the tracks built from them?

i'll compare standard bytetrack with a version using calibrated scores, including
controlled occlusions. i'll also explore an image-plane kalman filter and a
bearing-only extended kalman filter, defining each filter's state and observability
assumptions. tracking metrics will follow the identities and coverage available in
the annotations.

## keeping the comparisons useful

each experiment will save its configuration, data selection, model version and
environment alongside its results. i'll choose settings on development data,
keep calibration and evaluation sequences separate, and document metric definitions
and timing boundaries. repeated runs will help me check how stable the results are.

each post will link to the run behind its plot or clip, including failure cases
and an explanation of what i think caused them.

## reading alongside the code

- akyon et al., *slicing aided hyper inference and fine-tuning for small object detection*.
- jiang et al., *anti-uav: a large-scale benchmark for vision-based uav tracking*.
- zhao et al., the dut anti-uav benchmark paper.
- küppers et al., *beyond classification: directly training calibrated object detectors*.
- ovadia et al., *can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift*.
- geifman and el-yaniv, *selective classification for deep neural networks*.
- zhang et al., *bytetrack: multi-object tracking by associating every detection box*.

[back to the project](../README.md) · [data notes](../data/README.md)
