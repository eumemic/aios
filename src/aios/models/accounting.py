"""Inference accounting shared by sessions and workflow runs.

Accounting follows the immutable *creation* edge.  Invocation edges are
deliberately absent: calling an existing session does not transfer ownership of
that session's spend.  ``own`` is inference performed by the node itself;
``subtree`` is ``own`` plus every creation descendant, across both session and
workflow-run boundaries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

UsageNodeKind = Literal["session", "run"]
UsageMetric = Literal["cost_microusd", "total_tokens"]
DEFAULT_USAGE_WINDOW_SECONDS = 86_400
MIN_USAGE_WINDOW_SECONDS = 60
MAX_USAGE_WINDOW_SECONDS = 2_592_000


class UsageNodeRef(BaseModel):
    """One immutable parent in the creation-accounting tree."""

    kind: UsageNodeKind
    id: str


class UsageCounters(BaseModel):
    """Cumulative inference bought at one node or in one subtree."""

    cost_microusd: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageRate(BaseModel):
    """Rolling-window inference rate, normalized to one hour.

    The usage ledger starts when the accounting migration lands.  Until one
    full requested window has elapsed, ``complete`` is false and
    ``observed_seconds`` states the actual denominator.  Rates remain useful
    immediately without pretending pre-ledger history was observed.
    """

    window_seconds: int
    observed_seconds: int
    complete: bool
    cost_microusd_per_hour: float = 0.0
    input_tokens_per_hour: float = 0.0
    output_tokens_per_hour: float = 0.0
    cache_read_input_tokens_per_hour: float = 0.0
    cache_creation_input_tokens_per_hour: float = 0.0


class AttributedUsage(BaseModel):
    """Own and transitive usage for one node in the accounting tree."""

    own: UsageCounters = Field(default_factory=UsageCounters)
    subtree: UsageCounters = Field(default_factory=UsageCounters)
    own_rate: UsageRate | None = None
    subtree_rate: UsageRate | None = None


class UsageConsumer(BaseModel):
    """One root consumer in the ranked account-wide usage view."""

    rank: int
    kind: UsageNodeKind
    id: str
    label: str
    status: str
    created_at: datetime
    archived_at: datetime | None = None
    share: float
    usage: AttributedUsage


class UsageConsumersResponse(BaseModel):
    """Ranked, additive root consumers for one account."""

    metric: UsageMetric
    window_seconds: int
    coverage_started_at: datetime
    total_rate_per_hour: float
    items: list[UsageConsumer] = Field(default_factory=list)
