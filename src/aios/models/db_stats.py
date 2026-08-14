"""Read-only database storage statistics."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TableStorageStats(BaseModel):
    name: str
    total_bytes: int
    heap_bytes: int
    index_bytes: int
    toast_bytes: int
    row_estimate: int
    dead_tuple_estimate: int


class MonthlyStorageBucket(BaseModel):
    table: str
    month: str
    row_estimate: int
    approx_bytes: int


class DatabaseStats(BaseModel):
    generated_at: datetime
    database_bytes: int
    tables: list[TableStorageStats] = Field(default_factory=list)
    buckets: list[MonthlyStorageBucket] = Field(default_factory=list)
