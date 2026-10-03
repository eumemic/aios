"""The request a session composes, as bytes (#2471).

One encoding serves every request hash and every request blob, so a hash
computed when a request is sent and one computed when it is rebuilt agree
exactly when the requests do.

:func:`encode` is order-preserving (tool order and key order are part of the
request), keeps non-ASCII characters as UTF-8, and is total over everything a
request can carry: NUL characters and lone surrogates from tool output or MCP
schemas, and NaN or infinite floats. That's why it can't reuse
``aios.workflows.determinism.canonical_json``, which sorts keys and rejects
NUL, lone surrogates and NaN because its values must fit in jsonb.

:data:`RENDER_VERSION` names the renderer that turns the event log into a
request (``build_messages`` and ``finalize_messages``). Bump it with any change
that alters rendered bytes. ``tests/unit/test_render_golden.py`` renders a fixed
log to a pinned hash, so a rendering change can't land without a deliberate
bump. Such a change also busts every session's prompt cache once, so it
deserves the attention.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

RENDER_VERSION = 1


def encode(value: Any) -> bytes:
    """The bytes a request value is hashed and stored as."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8", "surrogatepass"
    )


def sha256_hex(value: Any) -> str:
    """The content address of ``value``: sha256 over :func:`encode`."""
    return hashlib.sha256(encode(value)).hexdigest()
