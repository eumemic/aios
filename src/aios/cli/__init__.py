"""aios client CLI.

The typer app in ``app.py`` wires every subcommand. Resource modules in
``commands/`` implement CRUD against the HTTP API through the generated
``aios_sdk`` client — typed ops for schema-stable endpoints, the raw-body
arm (:func:`aios_sdk.raw_request`) for schema-fluid ones.

Entry points:

* ``aios <subcommand>`` (via ``[project.scripts]`` in ``pyproject.toml``)
* ``python -m aios <subcommand>`` (via ``__main__.py``)
"""

from __future__ import annotations
