"""Resolve a ``[ref=eN]`` handle back to a live element.

Two facts, one owner each. The PAGE owns only the current snapshot's ref→element
map (``window.__aiosRefs = {gen, map}``, overwritten wholesale per snapshot; the
document — hence the registry — is destroyed by navigation). The DRIVER owns the
``issued`` watermark: refs are numbered sequentially across a page's snapshots,
so a ref numbered above ``issued`` was never minted (``no_such_ref``, decided
without a round trip), while one at-or-below ``issued`` that no longer resolves
was minted by an older, now-superseded snapshot (``stale_snapshot``). Both
missing cases share the same remedy — take a fresh snapshot.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from aios_browser_driver.errors import ActionError

if TYPE_CHECKING:
    from playwright.async_api import ElementHandle, Page

    from aios_browser_driver.host import PageEntry

_REF_RE = re.compile(r"^e[1-9][0-9]*$")

# Return the element only if it is from the CURRENT generation and still in the
# document; anything else (superseded generation, detached node, navigation
# cleared the registry) returns null → stale_snapshot.
_FETCH_JS = """([ref, gen]) => {
  const store = window.__aiosRefs;
  if (!store || store.gen !== gen) return null;
  const el = store.map[ref];
  return el && el.isConnected ? el : null;
}"""

_RETAKE = "take a fresh browser_snapshot and use its refs"


async def resolve_ref(page: Page, entry: PageEntry, ref: str) -> ElementHandle:
    """Return the live element for ``ref`` or raise the classified failure."""
    if not _REF_RE.match(ref):
        raise ActionError("no_such_ref", f"{ref!r} is not a snapshot ref (expected e.g. 'e12')")
    if int(ref[1:]) > entry.issued:
        raise ActionError("no_such_ref", f"{ref} was never issued by a snapshot of this page")
    handle = await page.evaluate_handle(_FETCH_JS, [ref, entry.generation])
    element = handle.as_element()
    if element is None:
        raise ActionError(
            "stale_snapshot",
            f"{ref} is from a superseded snapshot (the page was re-observed, "
            f"navigated, or restarted) — {_RETAKE}",
        )
    return element
