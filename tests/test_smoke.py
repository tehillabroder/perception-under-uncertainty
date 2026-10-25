"""
check run records, output preservation and command line errors.
"""

import json
import subprocess
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest

from perception_under_uncertainty import runner
from perception_under_uncertainty.cli import main
from perception_under_uncertainty.runner import run_smoke

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"


# check the saved settings and environment so someone can trace how the run was configured.
def test_smoke_manifest_records_effective_settings_and_environment(tmp_path: Path) -> None:
    manifest_path = run_smoke(REFERENCE_CONFIG, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["smoke_check"] == "passed"
    assert manifest["status"] == "scaffold_only"
    assert manifest["model_inference_performed"] is False
    assert manifest["dataset_evaluated"] is False
    assert manifest["seed"] == 42
    assert manifest["config_sha256"] == sha256(REFERENCE_CONFIG.read_bytes()).hexdigest()
    assert manifest["config"]["output_dir"] == "artifacts/smoke"
    assert manifest["effective_config"]["output_dir"] == str(tmp_path.resolve())
    assert manifest["output_override"] == str(tmp_path)
    assert manifest["run_id"] == manifest_path.parent.name
    assert manifest["run_directory"] == str(manifest_path.parent)
    assert datetime.fromisoformat(manifest["executed_at_utc"]).utcoffset().total_seconds() == 0
    assert manifest["environment"]["packages"]["pydantic"]
    assert manifest["environment"]["machine"]


# run twice in the same output root to check that earlier records survive.
def test_later_runs_preserve_earlier_records(tmp_path: Path) -> None:
    first_path = run_smoke(REFERENCE_CONFIG, tmp_path)
    first_record = first_path.read_bytes()
    second_path = run_smoke(REFERENCE_CONFIG, tmp_path)

    assert first_path != second_path
    assert first_path.read_bytes() == first_record
    assert second_path.is_file()


# check where relative paths start so outputs land in the documented location.
def test_relative_output_starts_at_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = run_smoke(REFERENCE_CONFIG)
    assert manifest_path.parent.parent == tmp_path / "artifacts" / "smoke"


# validate before creating folders so bad input leaves no empty run outputs.
def test_invalid_config_creates_no_output_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "outputs"
    with pytest.raises(ValueError):
        run_smoke(config_path, output_root)
    assert not output_root.exists()


# simulate a failed final file move so an incomplete run leaves no success record or clutter.
def test_failed_publish_leaves_no_manifest_or_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(runner.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated write failure"):
        run_smoke(REFERENCE_CONFIG, tmp_path)
    assert list(tmp_path.iterdir()) == []


# allow runs without git while clearly recording that the source details are unavailable.
def test_missing_git_is_recorded_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".git").mkdir()

    def missing_git(*arguments: object, **options: object) -> None:
        raise FileNotFoundError("git is not installed")

    def editable_source_root() -> Path:
        return source_root

    monkeypatch.delenv("PPU_SOURCE_REVISION", raising=False)
    monkeypatch.setattr(runner, "_source_root", editable_source_root)
    monkeypatch.setattr(runner.subprocess, "run", missing_git)
    manifest_path = run_smoke(REFERENCE_CONFIG, tmp_path / "outputs")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"] == {
        "revision": None,
        "dirty": None,
        "origin": "unavailable",
    }

# check build-supplied details so container runs can be traced to their source and environment.
def test_build_metadata_and_lockfile_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_bytes(b"example lock contents\n")
    monkeypatch.setenv("PPU_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("PPU_SOURCE_DIRTY", "true")
    monkeypatch.setenv("PPU_LOCKFILE", str(lockfile))
    monkeypatch.setenv("PPU_IMAGE_ID", "sha256:" + "b" * 64)
    manifest_path = run_smoke(REFERENCE_CONFIG, tmp_path / "outputs")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"] == {
        "revision": "a" * 40,
        "dirty": True,
        "origin": "build_metadata",
    }
    assert manifest["lockfile_sha256"] == sha256(lockfile.read_bytes()).hexdigest()
    assert manifest["environment"]["image_id"] == "sha256:" + "b" * 64

# record the commit and any uncommitted changes so local edits are visible in the run record.
@pytest.mark.parametrize("git_status", ["", " M src/example.py\n?? new-config.json\n"])
def test_checkout_revision_and_dirty_state_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git_status: str
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".git").mkdir()

    def editable_source_root() -> Path:
        return source_root

    def git_result(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        assert command[:3] == ["git", "-C", str(source_root)]
        if command[3:] == ["rev-parse", "HEAD"]:
            output = "a" * 40 + "\n"
        else:
            assert command[3:] == ["status", "--porcelain", "--untracked-files=normal"]
            output = git_status
        return subprocess.CompletedProcess(command, returncode=0, stdout=output, stderr="")

    monkeypatch.delenv("PPU_SOURCE_REVISION", raising=False)
    monkeypatch.setattr(runner, "_source_root", editable_source_root)
    monkeypatch.setattr(runner.subprocess, "run", git_result)
    manifest_path = run_smoke(REFERENCE_CONFIG, tmp_path / "outputs")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"] == {
        "revision": "a" * 40,
        "dirty": bool(git_status),
        "origin": "git",
    }

# check the success message includes the saved file's path so the user can find the record.
def test_cli_reports_the_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = main(["run", "--config", str(REFERENCE_CONFIG), "--output-dir", str(tmp_path)])
    assert status == 0
    output = capsys.readouterr()
    assert "smoke check passed" in output.out
    assert "no model evaluation" in output.out
    assert output.err == ""
    manifest_paths = list(tmp_path.glob("*/run_manifest.json"))
    assert len(manifest_paths) == 1
    assert str(manifest_paths[0]) in output.out

# check bad input produces an error and exit code 2 so users and scripts can detect failure.
@pytest.mark.parametrize("contents", [None, "{", "{}"])
def test_cli_reports_bad_input_without_claiming_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], contents: str | None
) -> None:
    config_path = tmp_path / "bad.json"
    if contents is not None:
        config_path.write_text(contents, encoding="utf-8")
    with pytest.raises(SystemExit) as failure:
        main(["run", "--config", str(config_path)])
    assert failure.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "configuration or run error" in output.err

# use a file where a directory is needed to check that output errors reach the user.
def test_cli_reports_an_unwritable_output_location(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.touch()
    with pytest.raises(SystemExit) as failure:
        main(["run", "--config", str(REFERENCE_CONFIG), "--output-dir", str(output_file)])
    assert failure.value.code == 2
    assert "configuration or run error" in capsys.readouterr().err

# launch from another directory to catch imports that accidentally depend on the project folder.
def test_installed_module_runs_outside_the_checkout(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "perception_under_uncertainty",
            "run",
            "--config",
            str(REFERENCE_CONFIG),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert "smoke check passed" in result.stdout
    assert len(list(tmp_path.glob("*/run_manifest.json"))) == 1