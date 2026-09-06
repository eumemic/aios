"""Environment model validation tests."""

import pytest
from pydantic import ValidationError

from aios.models.environments import EnvironmentConfig
from aios.sandbox.limits import MAX_BASH_TIMEOUT_SECONDS


def test_packages_rejects_unsupported_package_manager() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        EnvironmentConfig(packages={"pyp": ["requests"]})


@pytest.mark.parametrize(
    "timeout",
    [MAX_BASH_TIMEOUT_SECONDS - 1, MAX_BASH_TIMEOUT_SECONDS],
)
def test_bash_timeout_accepts_values_through_shared_maximum(timeout: int) -> None:
    assert EnvironmentConfig(bash_timeout_seconds=timeout).bash_timeout_seconds == timeout


def test_bash_timeout_rejects_value_above_shared_maximum() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        EnvironmentConfig(bash_timeout_seconds=MAX_BASH_TIMEOUT_SECONDS + 1)
