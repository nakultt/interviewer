import asyncio
from typing import Any
from uuid import UUID

from pymongo import MongoClient
from pymongo.collection import Collection

from .config import Settings
from .domain import InterviewRecord, InterviewReport


class InterviewRepository:
    def __init__(self, settings: Settings) -> None:
        self._memory: dict[UUID, InterviewRecord] = {}
        self._reports: dict[UUID, InterviewReport] = {}
        self._collection: Collection[dict[str, Any]] | None = None
        self._report_collection: Collection[dict[str, Any]] | None = None
        self._client: MongoClient[dict[str, Any]] | None = None
        if settings.mongodb_uri:
            self._client = MongoClient(
                settings.mongodb_uri.get_secret_value(),
                serverSelectionTimeoutMS=5_000,
                connectTimeoutMS=5_000,
                appname="KEC Local Interviewer",
                connect=False,
            )
            database = self._client[settings.mongodb_database]
            self._collection = database["interviews"]
            self._report_collection = database["reports"]

    async def ping(self) -> bool:
        if not self._client:
            return False
        try:
            await asyncio.to_thread(self._client.admin.command, "ping")
            return True
        except Exception:
            return False

    async def save(self, record: InterviewRecord) -> None:
        self._memory[record.id] = record.model_copy(deep=True)
        if self._collection is not None:
            document = record.model_dump(mode="json")
            await asyncio.to_thread(
                self._collection.replace_one,
                {"id": str(record.id)},
                document,
                True,
            )

    async def get(self, interview_id: UUID) -> InterviewRecord | None:
        record = self._memory.get(interview_id)
        if record:
            return record.model_copy(deep=True)
        if self._collection is not None:
            document = await asyncio.to_thread(
                self._collection.find_one, {"id": str(interview_id)}, {"_id": 0}
            )
            if document:
                parsed = InterviewRecord.model_validate(document)
                self._memory[interview_id] = parsed
                return parsed.model_copy(deep=True)
        return None

    async def save_report(self, report: InterviewReport) -> None:
        self._reports[report.interview_id] = report.model_copy(deep=True)
        if self._report_collection is not None:
            document = report.model_dump(mode="json")
            await asyncio.to_thread(
                self._report_collection.replace_one,
                {"interview_id": str(report.interview_id)},
                document,
                True,
            )

    async def get_report(self, interview_id: UUID) -> InterviewReport | None:
        report = self._reports.get(interview_id)
        if report:
            return report.model_copy(deep=True)
        if self._report_collection is not None:
            document = await asyncio.to_thread(
                self._report_collection.find_one,
                {"interview_id": str(interview_id)},
                {"_id": 0},
            )
            if document:
                return InterviewReport.model_validate(document)
        return None
