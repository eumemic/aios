"""Guard WORKFLOW_SCRIPT_CONTRACT against the injected author namespace.

``WORKFLOW_SCRIPT_CONTRACT`` (``aios.models.workflows``) is the model-facing
statement of what a workflow-author script may call without imports. The set of
capabilities actually injected into the author namespace lives in
``aios.workflows.wf_script_host.author_namespace``. The two must agree: an agent
authoring a workflow from the contract is otherwise told a capability set that
silently drifts from the one it's handed (this guard was added after ``phase()``
shipped into the namespace without ever entering the contract prose).

Modeled on ``tests/unit/test_agent_tooltype_registry_drift.py``: introspect the
runtime source of truth, subtract a known-exclusions set, and assert structural
coverage. Importing ``wf_script_host`` here is cheap and credential-free — it
imports only stdlib + ``aios.workflows._protocol`` + ``aios.workflows.determinism``;
the credential-free isolation constraint binds the spawned child subprocess, not
this test process.
"""

from __future__ import annotations

from aios.models.workflows import WORKFLOW_SCRIPT_CONTRACT
from aios.workflows.wf_script_host import author_namespace

# Names present in the injected namespace that are deliberately NOT named in the
# contract prose. ``__builtins__`` is plumbing, not a capability. The exception
# types are intentionally prose-abstracted in the contract ("tool errors are
# returned, not raised"; "a failed agent branch yields None") rather than named —
# an intentional omission, not drift.
KNOWN_EXCLUSIONS = {"__builtins__", "AgentError", "AgentNoReturnError"}


def test_contract_documents_every_injected_capability() -> None:
    """Every public injected capability name must appear in the contract prose."""
    injected = set(author_namespace()) - KNOWN_EXCLUSIONS

    missing = {name for name in injected if f"{name}(" not in WORKFLOW_SCRIPT_CONTRACT}

    assert not missing, (
        "WORKFLOW_SCRIPT_CONTRACT omits injected author capabilities: "
        f"{sorted(missing)}. Add a bullet for each to the 'Injected capability "
        "API' list (aios.models.workflows), or add to KNOWN_EXCLUSIONS if "
        "deliberately prose-abstracted."
    )
