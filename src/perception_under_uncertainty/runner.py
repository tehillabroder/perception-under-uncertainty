"""
save the settings, environment and source information for each smoke run.

each run gets its own folder. the manifest (run receipt) is written to a temporary file and
moved into place once it is complete.
"""

import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

from . import __version__
from .config import load_config


def _source_root() -> Path | None:
    """
    find the source checkout that contains this installed module.
    """
    package_dir = Path(__file__).resolve().parent
    for parent in package_dir.parents:
        if (parent / "src" / "perception_under_uncertainty") == package_dir:
            if (parent / "pyproject.toml").is_file():
                return parent
    return None


def _source_details(source_root: Path | None) -> dict[str, str | bool | None]:
    """
    record which git commit the run came from and whether it included uncommitted edits.

    containers get these details from environment variables because the image
    doesn't include the .git folder. local runs read them from the source checkout.
    """
    revision = os.environ.get("PPU_SOURCE_REVISION")
    if revision and revision != "unknown":
        dirty_value = os.environ.get("PPU_SOURCE_DIRTY", "unknown")
        dirty = {"true": True, "false": False}.get(dirty_value)
        return {"revision": revision, "dirty": dirty, "origin": "build_metadata"}

    unknown = {"revision": None, "dirty": None, "origin": "unavailable"}
    if source_root is None or not (source_root / ".git").exists():
        return unknown
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        status_result = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return unknown
    return {
        "revision": revision_result.stdout.strip() or None,
        "dirty": bool(status_result.stdout.strip()),
        "origin": "git",
    }


def _lock_hash(source_root: Path | None) -> str | None:
    """
    hash the lockfile from the checkout or the path supplied by the container.
    """
    lock_override = os.environ.get("PPU_LOCKFILE")
    if lock_override:
        lock_path = Path(lock_override)
    elif source_root is not None:
        lock_path = source_root / "uv.lock"
    else:
        return None
    if not lock_path.is_file():
        return None
    return sha256(lock_path.read_bytes()).hexdigest()


def _installed_packages() -> dict[str, str]:
    """
    record the package versions actually installed in this environment.
    """
    packages = {}
    for distribution in distributions():
        package_name = distribution.metadata["Name"]
        if package_name:
            packages[package_name] = distribution.version
    return dict(sorted(packages.items()))


def _write_manifest(manifest: dict[str, object], destination: Path) -> None:
    """
    write the finished manifest and clean up the temporary file on failure.
    """
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=".manifest-",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as manifest_file:
            temporary_path = Path(manifest_file.name)
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_smoke(config_path: Path, output_override: Path | None = None) -> Path:
    """
    validate the settings and return the path to the new run's manifest.

    relative paths start at the working directory. the output override chooses
    the parent folder for the run. file and validation errors pass to the caller.
    """
    config_path = config_path.resolve()
    config, config_hash = load_config(config_path)
    output_root = output_override if output_override is not None else config.output_dir
    output_root = output_root.resolve()
    source_root = _source_root()
    executed_at = datetime.now(UTC)
    run_id = f"{executed_at:%Y%m%dT%H%M%S%fZ}-{uuid4().hex}"
    run_dir = output_root / run_id

    effective_config = config.model_dump(mode="json")
    effective_config["output_dir"] = str(output_root)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": config.experiment,
        "status": "scaffold_only",
        "smoke_check": "passed",
        "model_inference_performed": False,
        "dataset_evaluated": False,
        "seed": config.seed,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "config": config.model_dump(mode="json"),
        "effective_config": effective_config,
        "output_override": str(output_override) if output_override is not None else None,
        "working_directory": str(Path.cwd()),
        "run_directory": str(run_dir),
        "source": _source_details(source_root),
        "lockfile_sha256": _lock_hash(source_root),
        "package_version": __version__,
        "environment": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "image_id": os.environ.get("PPU_IMAGE_ID"),
            "packages": _installed_packages(),
        },
        "executed_at_utc": executed_at.isoformat(),
    }

    # creating a new directory exclusively prevents another run being overwritten.
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    try:
        _write_manifest(manifest, manifest_path)
    except OSError:
        # leave unexpected files alone, but remove an empty failed run directory.
        if not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise
    return manifest_path
