# Perception Under Uncertainty: From raw pixels to real world

I'm building a series of small computer vision experiments to understand how
perception changes when images lose detail, cameras move, and sensors disagree.
i'll share the code, results and things that go wrong as i work through them.

## Post 1: Small objects and observability

The experiments are complete. [More pixels, more drones?](docs/post1_independent_evaluation.md) follows the question through 300 new images, with [saved measurements and visual cases](runs/dut_300_evaluation/README.md) behind the findings.

I started with 30 selected DUT Anti-UAV images, comparing whole-image YOLO at 640 and 1280 with SAHI sliced inference, then added a sliced-plus-full-image comparison.

In the 300-image follow-up, higher-resolution YOLO recovered more drones, including at similar false-positive counts. Slicing helped some targets, but produced less accurate boxes and more extra detections, and took substantially longer.

The [evaluation notes](docs/post1_xml_eval.md) and [published results](results/post1/README.md) record the earlier 30-image exploration, including the difference between visually plausible detections and boxes that matched the annotations.

The [baseline notes](docs/post1_baseline_notes.md) and [manual observations](docs/post1_sahi_observations.md) preserve those early checks. The [methodology](docs/post1_sahi_methodology.md) also covers the completed 300-image evaluation, run by [`evaluate_post1_sample.py`](scripts/evaluate_post1_sample.py).

## The research

The series covers small objects, motion cues, monocular geometry, rgb/thermal
fusion, detection calibration and tracking. calibration is the central question
for me: how much should we trust a detection when the image conditions change?

See the [experiment plan](docs/experiment_plan.md) for the questions and papers,
and the [data notes](data/README.md) for the datasets i'm planning to use.

## Try the setup

The project uses **python 3.12.13 and uv 0.12.19**. with
[uv installed](https://docs.astral.sh/uv/getting-started/installation/), run these
commands from the project folder:

```bash
uv sync --locked
uv run --locked ppu run --config configs/smoke.json
```

This creates the local `.venv` and runs a setup check. in vscode, select
`.venv/bin/python` as the interpreter.

Each run saves its settings, environment details and available source information
in a new folder. the command prints the path:

```text
artifacts/smoke/<run-id>/run_manifest.json
```

[`configs/smoke.json`](configs/smoke.json) contains the configuration version,
experiment name, seed and output directory. relative paths start at the working
directory. use `--output-dir` to choose a different output root, or
`uv run --locked ppu run --help` for help.

## Running in Docker

I'm developing on an intel mac and will run the model experiments in linux
containers. with colima, start it using `colima start` and select
`docker context use colima`.

Build the image:

```bash
docker buildx build --load --platform linux/amd64 \
  --tag perception-under-uncertainty:dev .
```

Run it and save the output on your machine:

```bash
mkdir -p artifacts/docker-smoke
image_id=$(docker image inspect --format '{{.Id}}' perception-under-uncertainty:dev)
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env PPU_IMAGE_ID="$image_id" \
  --mount "type=bind,source=$PWD/artifacts/docker-smoke,target=/results" \
  perception-under-uncertainty:dev
```

Outputs appear in `artifacts/docker-smoke/`. the image uses pinned build inputs
and the dependencies in `uv.lock`. you can stop the local vm with `colima stop`.

<details>
<summary>including the git revision in a container run</summary>

From a git checkout, build with the revision and current edit status:

```bash
source_revision=$(git rev-parse HEAD)
source_dirty=false
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  source_dirty=true
fi
docker buildx build --load --platform linux/amd64 \
  --build-arg SOURCE_REVISION="$source_revision" \
  --build-arg SOURCE_DIRTY="$source_dirty" \
  --tag perception-under-uncertainty:dev .
```

Use a clean, committed checkout for published runs. missing source information
is recorded as `null`.

</details>

## Finding your way around

- [`docs/`](docs/) contains the experiment plan, research notes, methodology and observations.
- [`scripts/test_tomsmail.py`](scripts/test_tomsmail.py) runs the whole-image YOLO baselines at 640 and 1280.
- [`scripts/test_sahi.py`](scripts/test_sahi.py) runs sliced inference with SAHI.
- [`scripts/evaluate_post1.py`](scripts/evaluate_post1.py) runs inference, prints the baseline summary and saves predictions.
- [`scripts/analyse_post1.py`](scripts/analyse_post1.py) investigates sizes, missed targets and errors from the saved cache.
- [`scripts/sweep_post1_confidence.py`](scripts/sweep_post1_confidence.py) measures confidence filtering and writes a csv.
- [`results/post1/`](results/post1/) contains the published cache, measurements and selected figures, with commands to reproduce the analysis.
- [`tests/`](tests/) checks configuration, run records, IoU calculations and detection matching.
- [`configs/smoke.json`](configs/smoke.json) contains the setup-check configuration.
- [`src/perception_under_uncertainty/`](src/perception_under_uncertainty/) handles configuration, the CLI and run records.
- `runs/` stores prediction caches and tables. The two frozen four-method exploration files listed in the [results guide](results/post1/README.md) and the selected [300-image evidence](runs/dut_300_evaluation/README.md) are explicitly included in Git; other run outputs stay local.
- `results/post1/figures/predictions/` keeps bulk detection visuals locally, alongside the published figures.

## Development checks

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest
```

[ci](.github/workflows/ci.yml) also checks the installed package and container.
to update a dependency, use `uv lock --upgrade-package <package>`, run
`uv sync --locked`, and repeat the checks.

Questions and suggestions are welcome as the experiments take shape.
The code is available under the [MIT license](LICENSE).
