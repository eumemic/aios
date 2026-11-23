"""Embedding of memory content for hybrid (FTS + vector) recall (#1629).

The embedding is a *derived index* of an already-versioned ``memories.content``
row — it never allocates a ``MemoryVersion``. Unlike ``content_tsv`` (migration
0119, ``GENERATED ... STORED``) an embedding needs an async model call, so it is
written explicitly at the service layer, AFTER the content transaction commits.

Write-failure policy (FORK a) is DECOUPLED and encoded as a KIND, never a flag:
:func:`embed_memory_content` returns :class:`Embedded` or :class:`Skipped`.
A ``Skipped`` outcome leaves ``embedding``/``embedding_model`` NULL — a
representable, queryable row state that the ``vec`` ranking arm simply
excludes (FTS still finds the row) and that ``bin/aios-memory-embed-backfill``
later fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg
import litellm

from aios.config import get_settings
from aios.logging import get_logger

log = get_logger(__name__)

# Immutable per column (migration 0196): ``vector(1536)``. A model whose output
# dimension differs is a re-embed migration, never a runtime switch.
EMBEDDING_DIM = 1536

# Reciprocal-Rank-Fusion constant (literature default). Mirrored as a literal in
# ``aios.tools.memory_search._SEARCH_SQL``.
RRF_K = 60


@dataclass(frozen=True)
class Embedded:
    vector: list[float]
    model: str


@dataclass(frozen=True)
class Skipped:
    reason: str


EmbedOutcome = Embedded | Skipped


def rrf_score(*ranks: int | None, k: int = RRF_K) -> float:
    """Reciprocal-rank fusion of 1-based ranks; ``None`` (absent from an arm) adds 0."""
    return sum(1.0 / (k + r) for r in ranks if r is not None)


def vector_literal(vector: list[float]) -> str:
    """pgvector text input form (``[a,b,...]``) — bound as ``$n::vector``."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


async def embed_memory_content(content: str) -> EmbedOutcome:
    """Embed ``content`` with ``settings.memory_embedding_model``.

    Never raises for provider trouble: any failure (no model configured,
    provider error, wrong dimension) is a :class:`Skipped` outcome.
    """
    model = get_settings().memory_embedding_model
    if not model:
        return Skipped("memory_embedding_model not configured")
    try:
        response: Any = await litellm.aembedding(model=model, input=[content])
        data = response["data"] if isinstance(response, dict) else response.data
        first = data[0]
        raw = first["embedding"] if isinstance(first, dict) else first.embedding
        vector = [float(x) for x in raw]
    except Exception as exc:
        log.warning("memory_embedding.failed", model=model, error=str(exc))
        return Skipped(f"embedding provider error: {exc}")
    if len(vector) != EMBEDDING_DIM:
        log.warning("memory_embedding.bad_dim", model=model, dim=len(vector))
        return Skipped(f"embedding dimension {len(vector)} != {EMBEDDING_DIM}")
    return Embedded(vector=vector, model=model)


async def embed_and_store(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    memory_id: str,
    content: str,
    content_sha256: str,
) -> EmbedOutcome:
    """Embed ``content`` and write it onto the memory row, guarded by content hash.

    The UPDATE only lands while the row still carries ``content_sha256`` — a
    racing content update that committed in between makes this a no-op (its own
    embed writes the fresh vector), so a stale embedding can never overwrite a
    newer one.
    """
    from aios.db import queries

    outcome = await embed_memory_content(content)
    if isinstance(outcome, Embedded):
        async with pool.acquire() as conn:
            await queries.update_memory_embedding(
                conn,
                memory_id=memory_id,
                embedding=outcome.vector,
                embedding_model=outcome.model,
                content_sha256=content_sha256,
                account_id=account_id,
            )
    else:
        log.info("memory_embedding.skipped", memory_id=memory_id, reason=outcome.reason)
    return outcome
