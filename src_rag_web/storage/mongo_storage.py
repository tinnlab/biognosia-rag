"""Async MongoDB adapter for paper metadata lookups."""

import logging
from typing import Any

from .._redaction import redact_credentials

logger = logging.getLogger(__name__)


class MongoStorage:
    """
    Async MongoDB adapter for retrieving paper metadata.

    Used by the citation resolver to fetch paper title, authors, year,
    journal, DOI and PDF link for each cited chunk before the final answer
    is returned.

    Expected document schema (collection: papers):
        paperId:          str   — 40-char hex hash matching chunk ID prefix
        title:            str
        bibtex_json:      dict  — {authors: list[str], year: int}
        publicationDate:  str
        journal:          dict  — {name: str}
        doi:              str
        pdf_url:          str
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._client = None
        self._collection = None

    async def initialize(self):
        try:
            import motor.motor_asyncio  # type: ignore[import]
        except ImportError as e:
            raise ImportError("motor is required for MongoDB support: pip install motor") from e

        uri = self._config.get("uri", "mongodb://localhost:27017")
        database = self._config.get("database", "paper_db")
        collection_name = self._config.get("collection", "papers")

        self._client = motor.motor_asyncio.AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        db = self._client[database]
        self._collection = db[collection_name]

        # Verify connectivity
        await self._client.admin.command("ping")
        logger.info(f"MongoDB connected: {redact_credentials(uri)} / {database}.{collection_name}")

    async def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None

    async def get_papers_metadata(self, paper_ids: list[str]) -> dict[str, dict]:
        """
        Batch-fetch paper metadata for the given paper IDs.

        Returns a dict keyed by paperId. Missing papers are simply absent
        from the result — callers should handle that gracefully.
        """
        if not paper_ids or self._collection is None:
            return {}

        unique_ids = list(set(paper_ids))
        cursor = self._collection.find(
            {"paperId": {"$in": unique_ids}},
            {
                "paperId": 1,
                "title": 1,
                "bibtex_json": 1,
                "publicationDate": 1,
                "journal": 1,
                "doi": 1,
                "pdf_url": 1,
                "_id": 0,
            },
        )

        result: dict[str, dict] = {}
        async for doc in cursor:
            pid = doc.get("paperId")
            if pid:
                result[pid] = doc

        return result
