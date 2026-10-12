"""Service-layer pin for the no-default ``reader`` property on run listing (#2537).

``db/queries/workflows.list_wf_runs`` has no default ``reader`` (pinned in
``tests/unit/db/test_wf_runs_recency_order.py``). The service wrapper the agent
tools call must not reintroduce one either: a caller that omits ``reader`` is
refused before any DB work rather than handed the unfiltered operator view.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aios.services import workflows as wf_service


@pytest.mark.asyncio
async def test_list_runs_has_no_default_reader() -> None:
    pool = MagicMock()
    with pytest.raises(TypeError, match="reader"):
        await wf_service.list_runs(pool, account_id="acc_1")  # type: ignore[call-arg]
    pool.acquire.assert_not_called()
