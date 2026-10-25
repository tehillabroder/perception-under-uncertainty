# data

i'll start with small subsets of two datasets:

- **dut anti-uav** for small-object detection and motion experiments.
- **anti-uav300** for paired rgb and thermal sequences in the fusion experiment.

as i inspect each release, i'll document how to get it, which samples i'm using,
and how its annotations work. for the paired videos, i'll also check how well the
two streams line up in space and time.

downloads stay in this folder and are ignored by git. i'll keep the raw files
unchanged, save subset selections and preprocessing settings in `configs/`, and
write generated outputs to `artifacts/`.

each subset will have a source, version, checksum and license recorded. video
splits will use separate sequences for development, calibration and evaluation.
i'll check the dataset's sharing terms before including frames in public posts.

[back to the project](../README.md) · [experiment plan](../docs/experiment_plan.md)