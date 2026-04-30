"""System-prompt block describing attached memory stores.

Mounted memory stores are otherwise invisible to the model — the agent has
no special "memory tool", just regular file access at
``/mnt/memory/<store_name>/``. The prompt block tells it where each store
lives, whether it can write, what the store is for, and any session-specific
instructions the caller attached.
"""

from __future__ import annotations

from aios.models.memory_stores import MemoryStoreResourceEcho


def build_memory_stores_block(echoes: list[MemoryStoreResourceEcho]) -> str:
    """Render the system-prompt section for attached memory stores.

    Returns the empty string when no stores are attached so the augmenter
    can keep the system prompt unchanged in the common case.
    """
    if not echoes:
        return ""
    sections: list[str] = ["## Memory stores"]
    for echo in echoes:
        lines = [
            f"### {echo.name}",
            f"- mount: `{echo.mount_path}`",
            f"- access: {echo.access}",
        ]
        if echo.description:
            lines.append(f"- description: {echo.description}")
        if echo.instructions:
            lines.append(f"- instructions: {echo.instructions}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def augment_with_memory_stores(base_system: str, echoes: list[MemoryStoreResourceEcho]) -> str:
    """Append the memory-stores block to ``base_system`` if any are attached."""
    block = build_memory_stores_block(echoes)
    if not block:
        return base_system
    if base_system:
        return base_system + "\n\n" + block
    return block
