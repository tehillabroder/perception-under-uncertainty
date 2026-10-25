"""
read and validate the settings for a smoke run.

unknown fields are rejected. relative output paths start at the working directory,
so the configuration works from the project folder or inside a container.
"""

from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    """
    describe the settings for one setup check.
    """

    # reject unknown fields and prevent reassignment after validation
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    experiment: Literal["smoke"]
    # must be an int within range of unsigned 32-bit
    seed: int = Field(strict=True, ge=0, le=2**32 - 1)
    output_dir: Path

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        """
        require an integer version, since booleans and floats can also equal one.
        """
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("output_dir", mode="before")
    @classmethod
    def reject_empty_output(cls, value: object) -> object:
        """
        catch empty paths before they become the working directory.
        """
        if isinstance(value, str) and not value.strip():
            raise ValueError("output_dir must not be empty")
        return value


def load_config(path: Path) -> tuple[RunConfig, str]:
    """
    return the validated settings and the hash of the bytes read.

    reading once keeps the hash tied to the settings if the file changes later.
    file and validation errors pass back to the caller.
    """
    config_bytes = path.read_bytes()
    config = RunConfig.model_validate_json(config_bytes)
    # the fingerprint of the exact config file contents
    return config, sha256(config_bytes).hexdigest()
