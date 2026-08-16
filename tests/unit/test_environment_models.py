"""Environment model validation tests."""

import pytest
from pydantic import ValidationError

from aios.models.environments import EnvironmentConfig


def test_packages_rejects_unsupported_package_manager() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        EnvironmentConfig(packages={"pyp": ["requests"]})
