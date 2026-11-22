"""A SQLite checkpointer, so a conversation (thread) survives restarts of the kernel.

LangGraph saves the graph state after every step, keyed by `thread_id`. With the in-memory
saver that history disappears when the process ends. This one keeps it in
solution/data/core/checkpoints.db (a separate file from udahub.db, so resetting the support
database does not wipe conversations).

    checkpointer = await open_checkpointer()
    orchestrator = await build_orchestrator(toolbox, checkpointer=checkpointer)
    ...
    await close_checkpointer(checkpointer)
"""

from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# solution/agentic/checkpointing.py -> parents[1] == solution/
CHECKPOINT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "core" / "checkpoints.db"


async def open_checkpointer(db_path: str | Path | None = None) -> AsyncSqliteSaver:
    """Open (and create if needed) the checkpoint database."""
    path = Path(db_path) if db_path else CHECKPOINT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = AsyncSqliteSaver(await aiosqlite.connect(str(path)))
    await saver.setup()
    return saver


async def close_checkpointer(saver: AsyncSqliteSaver) -> None:
    await saver.conn.close()
