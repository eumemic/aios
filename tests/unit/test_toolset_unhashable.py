import pytest
from pydantic import ValidationError

from aios.models.agents import AgentCreate, AgentUpdate


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "toolset", "name": ["goal_management"], "version": 1},
        {"type": "toolset", "name": "goal_management", "version": {"a": 1}},
        {"type": "toolset", "name": {"x": 1}, "version": [1]},
        {"type": "toolset"},
    ],
)
def test_malformed_toolset_is_validation_error(bad):
    with pytest.raises(ValidationError):
        AgentCreate.model_validate({"name": "a", "model": "m", "tools": [bad]})
    with pytest.raises(ValidationError):
        AgentUpdate.model_validate({"tools": [bad]})
