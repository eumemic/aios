"""``aios init`` — scaffold a typed agent-authoring project skeleton.

The inverse of :func:`aios.cli.files.walk_skill_dir`: instead of walking a
directory and reading its files for upload, ``init`` *writes* a known set of
files so a new author starts from a typed, working ``agent.py`` rather than
hand-writing raw JSON.

It reuses ``walk_skill_dir``'s fail-hard + symlink-reject stance: a symlinked
target is refused (a symlink could redirect writes outside the author's
intended tree — the same confused-deputy concern), and a non-empty target is
refused (re-running ``init`` over an existing project must not clobber edits).
In both cases nothing is written.

No server route, no ``operationId``, no network — this is a pure-local CLI
command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aios.cli.runtime import run_or_die

_AGENT_PY = '''\
"""A typed agent definition.

Edit this file to shape your agent, then deploy it (see README.md).
"""

from __future__ import annotations

from aios_sdk.authoring import define_agent, define_tool_builtin

AGENT = (
    define_agent("my-agent", "anthropic/claude-opus-4-6")
    .system("You are a helpful assistant.")
    .tool(define_tool_builtin("bash"))
    .build()
)


if __name__ == "__main__":
    import json

    print(json.dumps(AGENT, indent=2))
'''

_README = """\
# my-agent

A typed agent-authoring project scaffolded by `aios init`.

Edit `agent.py` to shape your agent: change the model, system prompt, and the
tools it can call. The `AGENT` value is built from the typed `aios_sdk.authoring`
facade, so your editor and type checker catch mistakes before you deploy.

## Deploy

Render the agent definition and create it on the server:

```sh
python agent.py | aios agents create --file -
```

Once `aios agents watch` lands (#1354) you will be able to keep the deployed
agent in sync with the file automatically:

```sh
aios agents watch agent.py
```
"""

_PYPROJECT = """\
[project]
name = "my-agent"
version = "0.1.0"
description = "A typed agent-authoring project."
requires-python = ">=3.11"
dependencies = [
    "aios-sdk",
]
"""


def register(app: typer.Typer) -> None:
    @app.command("init", help="Scaffold a typed agent-authoring project skeleton.")
    def init(
        dir_: Annotated[
            Path,
            typer.Argument(metavar="DIR", help="Target directory to scaffold."),
        ],
    ) -> None:
        def _run() -> int | None:
            target = dir_
            # Reject a symlinked target for the same confused-deputy reason
            # walk_skill_dir rejects symlinks (files.py:131-133): a symlinked
            # DIR could redirect writes outside the author's intended tree.
            if target.is_symlink():
                typer.secho(
                    f"refusing to scaffold into a symlink: {target}",
                    fg="red",
                    err=True,
                )
                return 1
            # No-clobber: if the dir exists and is non-empty, write NOTHING
            # (fail-hard, mirrors the PayloadError raises in files.py:111-117).
            if target.exists() and any(target.iterdir()):
                typer.secho(
                    f"refusing to scaffold into a non-empty directory: {target}",
                    fg="red",
                    err=True,
                )
                return 1
            target.mkdir(parents=True, exist_ok=True)
            (target / "agent.py").write_text(_AGENT_PY, encoding="utf-8")
            (target / "README.md").write_text(_README, encoding="utf-8")
            (target / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
            typer.secho(
                f"scaffolded agent-authoring project at {target}",
                fg="green",
            )
            return None

        run_or_die(_run)
