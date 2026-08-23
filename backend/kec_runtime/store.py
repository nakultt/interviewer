import asyncio
from collections import defaultdict
from pathlib import Path
from uuid import UUID

from .models import InterviewEvent


class EventStore:
    """Append-only NDJSON store; one lock per session prevents interleaved writes."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._locks: defaultdict[UUID, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def append(self, event: InterviewEvent) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / f"{event.session_id}.ndjson"
        line = event.model_dump_json() + "\n"
        async with self._locks[event.session_id]:
            await asyncio.to_thread(self._append_sync, path, line)

    @staticmethod
    def _append_sync(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
