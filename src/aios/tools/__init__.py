"""Aios v1 built-in tools.

Importing this package triggers registration of every built-in tool
against the module-level :data:`aios.tools.registry.registry` singleton.
The worker imports this module once at startup (via ``aios.harness.worker``),
after which :func:`aios.tools.registry.to_openai_tools` can translate any
agent's ``tools`` list into a LiteLLM ``tools`` parameter.

Phase 3 ships only the bash tool. Phase 4 adds read, write, edit, glob,
grep via vendored hermes ``ShellFileOperations``. Each new tool is a
side-effect import below — keep the list short and well-ordered.
"""

from __future__ import annotations

# Side-effect imports: each module's top-level _register() call adds the
# tool to the registry singleton. Order doesn't matter semantically but
# matches the agent-tools declaration order for readability.
from aios.tools import bash as _bash  # noqa: F401
