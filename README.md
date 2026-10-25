# Perception Under Uncertainty: From Pixels to 3D State Estimation

i'm building a series of small computer vision experiments to understand how
perception changes when images lose detail, cameras move, and sensors disagree.
i'll share the code, results and things that go wrong as i work through them.

the first experiment will compare resizing a whole image with running detection
on smaller patches. i'm interested in what happens to tiny objects, false alarms
and processing time.

**where things stand:** the configuration, run records, tests and container setup
are working. the perception experiments are the next step.

## the research

the series covers small objects, motion cues, monocular geometry, rgb/thermal
fusion, detection calibration and tracking. calibration is the central question
for me: how much should we trust a detection when the image conditions change?

see the [experiment plan](docs/experiment_plan.md) for the questions and papers,
and the [data notes](data/README.md) for the datasets i'm planning to use.

## try the setup

the project uses **python 3.12.13 and uv 0.12.19**. with
[uv installed](https://docs.astral.sh/uv/getting-started/installation/), run these
commands from the project folder:

```bash
uv sync --locked
uv run --locked ppu run --config configs/smoke.json
```

this creates the local `.venv` and runs a setup check. in vscode, select
`.venv/bin/python` as the interpreter.

each run saves its settings, environment details and available source information
in a new folder. the command prints the path:

```text
artifacts/smoke/<run-id>/run_manifest.json
```

[`configs/smoke.json`](configs/smoke.json) contains the configuration version,
experiment name, seed and output directory. relative paths start at the working
directory. use `--output-dir` to choose a different output root, or
`uv run --locked ppu run --help` for help.

## running in docker

i'm developing on an intel mac and will run the model experiments in linux
containers. with colima, start it using `colima start` and select
`docker context use colima`.

build the image:

```bash
docker buildx build --load --platform linux/amd64 \
  --tag perception-under-uncertainty:dev .
```

run it and save the output on your machine:

```bash
mkdir -p artifacts/docker-smoke
image_id=$(docker image inspect --format '{{.Id}}' perception-under-uncertainty:dev)
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env PPU_IMAGE_ID="$image_id" \
  --mount "type=bind,source=$PWD/artifacts/docker-smoke,target=/results" \
  perception-under-uncertainty:dev
```

outputs appear in `artifacts/docker-smoke/`. the image uses pinned build inputs
and the dependencies in `uv.lock`. you can stop the local vm with `colima stop`.

<details>
<summary>including the git revision in a container run</summary>

from a git checkout, build with the revision and current edit status:

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

use a clean, committed checkout for published runs. missing source information
is recorded as `null`.

</details>

## finding your way around

- [`config.py`](src/perception_under_uncertainty/config.py) reads and validates the settings.
- [`runner.py`](src/perception_under_uncertainty/runner.py) creates the run record.
- [`cli.py`](src/perception_under_uncertainty/cli.py) handles the command line.
- [`experiments/`](src/perception_under_uncertainty/experiments/) is where the research code will go.
- [`tests/`](tests/) checks configuration, run records and command behavior.

## development checks

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest
```

[ci](.github/workflows/ci.yml) also checks the installed package and container.
to update a dependency, use `uv lock --upgrade-package <package>`, run
`uv sync --locked`, and repeat the checks.

questions and suggestions are welcome as the experiments take shape.
the code is available under the [MIT license](LICENSE).