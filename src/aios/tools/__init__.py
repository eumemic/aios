"""Aios v1 built-in tools.

Importing this package triggers registration of every built-in tool
against the module-level :data:`aios.tools.registry.registry` singleton.
The worker imports this module once at startup (via ``aios.harness.worker``),
after which :func:`aios.tools.registry.to_openai_tools` can translate any
agent's ``tools`` list into a LiteLLM ``tools`` parameter.

Phase 3 shipped the bash tool. Phase 4 adds read, write, edit, and
search on top of vendored hermes ``ShellFileOperations``. Each tool is
a side-effect import below — keep the list short and well-ordered.
"""

from __future__ import annotations

# Side-effect imports: each module's top-level _register() call adds the
# tool to the registry singleton. Order doesn't matter semantically but
# matches the agent-tools declaration order for readability.
from aios.tools import bash as _bash  # noqa: F401
from aios.tools import edit as _edit  # noqa: F401
from aios.tools import read as _read  # noqa: F401
from aios.tools import search as _search  # noqa: F401
from aios.tools import write as _write  # noqa: F401
