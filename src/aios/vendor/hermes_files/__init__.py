"""Vendored hermes-agent file tool implementations.

See ``PATCHES.md`` in this directory for the vendoring provenance and
the complete list of changes applied on top of the upstream commit. The
public API consists of:

- :class:`~aios.vendor.hermes_files.file_operations.ShellFileOperations`
  — the async class that every aios file tool handler delegates to.
- :class:`~aios.vendor.hermes_files.file_operations.ReadResult`,
  :class:`~aios.vendor.hermes_files.file_operations.WriteResult`,
  :class:`~aios.vendor.hermes_files.file_operations.PatchResult`,
  :class:`~aios.vendor.hermes_files.file_operations.SearchResult` —
  the result dataclasses each tool handler wraps.
- :func:`~aios.vendor.hermes_files.fuzzy_match.fuzzy_find_and_replace`
  — the 8-strategy fuzzy find-and-replace used by the edit tool.
- :func:`~aios.vendor.hermes_files.binary_extensions.has_binary_extension`
  — the binary-file extension check used by the read tool.

The vendor dir is considered "aios internals" at this point — mypy
strict applies, the files are covered by the aios test suite, and
future changes should land as normal aios edits with updates to
``PATCHES.md`` documenting any further divergence from upstream.
"""

from __future__ import annotations
