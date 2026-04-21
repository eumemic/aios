"""Output formatting for the CLI.

Two default renderings:

* **Tables** for list responses — hand-rolled, no ``tabulate`` dependency.
* **Pretty JSON** for single resources and anything the user asks for with
  ``--format json``.

Color is emitted via raw ANSI escapes only when stdout is a TTY.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from typing import Any, Literal

OutputFormat = Literal["table", "json"]

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_CYAN = "\x1b[36m"


def _tty(stream: Any) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def color(text: str, code: str, *, stream: Any = sys.stdout) -> str:
    """Wrap ``text`` in an ANSI color only if ``stream`` is a TTY."""
    if not _tty(stream):
        return text
    return f"{code}{text}{_RESET}"


def dim(text: str, *, stream: Any = sys.stdout) -> str:
    return color(text, _DIM, stream=stream)


def bold(text: str, *, stream: Any = sys.stdout) -> str:
    return color(text, _BOLD, stream=stream)


def red(text: str, *, stream: Any = sys.stdout) -> str:
    return color(text, _RED, stream=stream)


def green(text: str, *, stream: Any = sys.stdout) -> str:
    return color(text, _GREEN, stream=stream)


def yellow(text: str, *, stream: Any = sys.stdout) -> str:
    return color(text, _YELLOW, stream=stream)


def cyan(text: str, *, stream: Any = sys.stdout) -> str:
    return color(text, _CYAN, stream=stream)


def print_json(obj: Any) -> None:
    """Write ``obj`` as pretty-printed JSON to stdout."""
    sys.stdout.write(json.dumps(obj, indent=2, sort_keys=False, default=_default))
    sys.stdout.write("\n")


def _default(value: Any) -> Any:
    # Fallback serializer for objects like datetimes that sneak through.
    try:
        return str(value)
    except Exception:
        return repr(value)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, default=_default)


def _truncate(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def render_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    headers: Sequence[str] | None = None,
    max_widths: dict[str, int] | None = None,
) -> str:
    """Render ``rows`` as a simple aligned table.

    ``columns`` are dict keys to pull from each row. ``headers`` is an
    optional display-name list matching ``columns`` length; defaults to
    uppercased column names. ``max_widths`` caps per-column width (values
    longer than the cap are truncated with an ellipsis).
    """
    headers = list(headers) if headers is not None else [c.upper() for c in columns]
    assert len(headers) == len(columns)

    max_widths = max_widths or {}
    rendered: list[list[str]] = []
    for row in rows:
        rendered.append([_stringify(row.get(c)) for c in columns])

    widths: list[int] = []
    for i, col in enumerate(columns):
        values = [headers[i]] + [r[i] for r in rendered]
        w = max(len(v) for v in values)
        cap = max_widths.get(col)
        if cap is not None:
            w = min(w, cap)
        widths.append(w)

    def fmt_row(cells: Iterable[str]) -> str:
        padded = [_truncate(c, widths[i]).ljust(widths[i]) for i, c in enumerate(cells)]
        return "  ".join(padded).rstrip()

    lines = [fmt_row(headers), fmt_row("-" * w for w in widths)]
    for r in rendered:
        lines.append(fmt_row(r))
    return "\n".join(lines) + "\n"


def print_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    headers: Sequence[str] | None = None,
    max_widths: dict[str, int] | None = None,
    empty_message: str = "(none)",
) -> None:
    """Print ``rows`` as a table. Emits ``empty_message`` when ``rows`` is empty."""
    if not rows:
        sys.stdout.write(dim(empty_message) + "\n")
        return
    sys.stdout.write(render_table(rows, columns, headers=headers, max_widths=max_widths))


def print_error(message: str) -> None:
    """Write ``message`` to stderr, red-tinted if attached to a TTY."""
    sys.stderr.write(red(f"error: {message}", stream=sys.stderr) + "\n")


def print_note(message: str) -> None:
    """Write a dim informational note to stderr (so it doesn't pollute stdout pipes)."""
    sys.stderr.write(dim(message, stream=sys.stderr) + "\n")


def print_success(verb: str, resource_id: str) -> None:
    """Write a ``<verb> <id>`` confirmation line to stdout for terminal verbs.

    Used by archive/delete commands whose server returns no body: we still
    want to give the user a visible acknowledgement and something scripts
    can grep for.
    """
    sys.stdout.write(f"{green(verb)} {resource_id}\n")
