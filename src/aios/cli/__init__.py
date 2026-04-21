"""aios client CLI.

The typer app in ``app.py`` wires every subcommand. Resource modules in
``commands/`` implement CRUD against the HTTP API via :class:`AiosClient`.

Entry points:

* ``aios <subcommand>`` (via ``[project.scripts]`` in ``pyproject.toml``)
* ``python -m aios <subcommand>`` (via ``__main__.py``)
"""

from __future__ import annotations
