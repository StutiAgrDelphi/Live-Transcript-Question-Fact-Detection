# component2/dedup.py
import os
import re
import math
import logging
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional

from openai import AsyncAzureOpenAI
from shared.schema import Flag

log = logging.getLogger(__name__)


@dataclass
class _DedupRecord:
    normalized_text: str
    embedding: Optional[List[float]]
    flag: Flag
    mention_count: int = 1


class DedupEngine:
    """
    Semantic dedup: two flags of the SAME type are duplicates if their
    resolved_text embeddings are cosine-similar above `threshold`.

    Comparison is scoped per flag.type on purpose — "did revenue grow?" and
    "revenue grew 12%" are semantically close but must never be deduped against
    each other; a question and its answer are both worth keeping.

    Falls back to exact-normalized-text matching if the embedding call fails,
    so a transient API error never silently lets an obvious repeat through.
    """

    def __init__(self, threshold: float = 0.90, max_records_per_type: int = 40):
        self.threshold = threshold
        self.max_records_per_type = max_records_per_type
        self.records: Dict[str, deque] = {}
        self._client: Optional[AsyncAzureOpenAI] = None
        self._embed_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self._embedding_failures = 0

    def _get_client(self) -> AsyncAzureOpenAI:
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
                api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
            )
        return self._client

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    async def _embed(self, text: str) -> Optional[List[float]]:
        try:
            client = self._get_client()
            resp = await client.embeddings.create(model=self._embed_deployment, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            self._embedding_failures += 1
            if self._embedding_failures <= 3:
                log.error(
                    f"Embedding call failed ({self._embedding_failures}/3) — dedup is "
                    f"running on exact-text-match fallback only, near-duplicate facts "
                    f"WILL slip through: {e}"
                )
            return None

    async def is_duplicate(self, flag: Flag) -> bool:
        text = (flag.resolved_text or flag.text).strip()
        if not text:
            return False

        norm_text = self._normalize(text)
        bucket = self.records.setdefault(flag.type, deque(maxlen=self.max_records_per_type))
        embedding = await self._embed(text)

        for record in bucket:
            if embedding is not None and record.embedding is not None:
                similarity = self._cosine(embedding, record.embedding)
            else:
                similarity = 1.0 if norm_text == record.normalized_text else 0.0
            if similarity >= self.threshold:
                record.mention_count += 1
                return True

        bucket.append(_DedupRecord(normalized_text=norm_text, embedding=embedding, flag=flag))
        return False


    async def verify(self) -> bool:
        """Call once when a session starts. Confirms the embedding deployment
        actually works, so a bad .env fails loudly instead of silently degrading
        dedup for the whole meeting."""
        test = await self._embed("startup check")
        if test is None:
            log.error(
                "DedupEngine startup check FAILED — semantic dedup is not working. "
                "Only exact repeats will be caught until this is fixed. Check "
                "AZURE_OPENAI_EMBEDDING_* in .env."
            )
            return False
        log.info("DedupEngine startup check passed — semantic dedup is active.")
        return True