"""
check configuration validation, required fields and input hashing.
"""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from perception_under_uncertainty.config import RunConfig, load_config

REFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"


@pytest.fixture
def settings() -> dict[str, object]:
    return json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))


# check the supplied example loads correctly so the quick-start command stays usable.
def test_reference_config_validates() -> None:
    config, config_hash = load_config(REFERENCE_CONFIG)
    assert config.experiment == "smoke"
    assert config.seed == 42
    assert config_hash == sha256(REFERENCE_CONFIG.read_bytes()).hexdigest()


# reject invalid seeds so a mistyped value cannot silently become a different setting.
@pytest.mark.parametrize("seed", ["42", True, 1.5, -1, 2**32])
def test_invalid_seed_is_rejected(settings: dict[str, object], seed: object) -> None:
    settings["seed"] = seed
    with pytest.raises(ValidationError, match="seed"):
        RunConfig.model_validate(settings)


# accept both ends of the allowed seed range to catch off-by-one validation mistakes.
@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_boundaries_are_accepted(settings: dict[str, object], seed: int) -> None:
    settings["seed"] = seed
    assert RunConfig.model_validate(settings).seed == seed


# reject unknown fields and unsupported values so configuration mistakes surface early.
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unsupported", True),
        ("experiment", "unimplemented_detection"),
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", "1"),
        ("output_dir", ""),
        ("output_dir", "   "),
    ],
)
def test_invalid_fields_are_rejected(
    settings: dict[str, object], field: str, value: object
) -> None:
    settings[field] = value
    with pytest.raises(ValidationError, match=field):
        RunConfig.model_validate(settings)


# require every setting so an incomplete configuration cannot start a run.
@pytest.mark.parametrize("field", ["schema_version", "experiment", "seed", "output_dir"])
def test_required_fields_cannot_be_omitted(settings: dict[str, object], field: str) -> None:
    del settings[field]
    with pytest.raises(ValidationError, match=field):
        RunConfig.model_validate(settings)


# preserve the missing-file error so the caller can explain why loading failed.
def test_missing_file_raises_a_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


# reject broken json so a damaged configuration cannot be used for a run.
def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError, match="Invalid JSON"):
        load_config(config_path)


# read the file once so its recorded hash matches the settings even if it changes later.
def test_validation_and_hashing_use_the_same_read(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, object]
) -> None:
    original_bytes = json.dumps(settings).encode()
    reads = []

    def read_once(path: Path) -> bytes:
        reads.append(path)
        if len(reads) > 1:
            raise AssertionError("the configuration was read more than once")
        return original_bytes

    monkeypatch.setattr(Path, "read_bytes", read_once)
    config, config_hash = load_config(REFERENCE_CONFIG)
    assert config.seed == settings["seed"]
    assert config_hash == sha256(original_bytes).hexdigest()
    assert reads == [REFERENCE_CONFIG]
