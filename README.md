# Perception Under Uncertainty: From Raw Pixels to Real World

I'm building a series of small computer vision experiments to understand how
perception changes when images lose detail, cameras move, and sensors disagree.
i'll share the code, results and things that go wrong as i work through them.

## Post 1: Small objects and observability

I'm comparing whole-image detection at 640 and 1280 with SAHI sliced inference on 30 selected images from DUT Anti-UAV.

So far, increasing resolution has recovered some missed drones, but also introduced more false positives. SAHI has been particularly noisy around trees, buildings and other background details. Some difficult drones are still missed regardless of resolution.

These are early manual observations, not final results. I'm now in the middle of checking predictions against the XML annotations to measure recall, false positives and localisation accuracy properly.

The [baseline notes](docs/post1_baseline_notes.md), [SAHI methodology](docs/post1_sahi_methodology.md) and [observations](docs/post1_sahi_observations.md) document what i've tested and found so far. The inference code is in [`test_tomsmail.py`](scripts/test_tomsmail.py) and [`test_sahi.py`](scripts/test_sahi.py).

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

## Running in docker

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
- [`scripts/evaluate_post1.py`](scripts/evaluate_post1.py) reads the DUT annotations and evaluates bounding-box matching.
- [`tests/`](tests/) checks configuration, run records, IoU calculations and detection matching.
- [`configs/smoke.json`](configs/smoke.json) contains the setup-check configuration.
- [`src/perception_under_uncertainty/`](src/perception_under_uncertainty/) handles configuration, the CLI and run records.
- `runs/` stores generated detection results locally and is excluded from git.

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
