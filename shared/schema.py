"""
shared/schema.py

This is the contract between Component 1 (transcript source) and
Component 2 (detection agent). Both sides — and both Claudes — should
import from here rather than redefining it locally, so the two
components can be built independently and still plug together.
"""

from dataclasses import dataclass, field
import uuid
from typing import Optional, List

@dataclass
class TranscriptChunk:
    """One unit of transcript delivered to the detection agent.

    Mirrors what a real live-captioning source (Teams, Azure Speech,
    etc.) would emit per utterance/sentence — Component 2 should be
    written against this shape and never know whether it came from a
    file, a websocket, or a speech-to-text pipeline.
    """

    speaker: str
    text: str
    elapsed_seconds: float          # seconds since meeting start — always populated, used for pacing/ordering
    is_final: bool = True           # live captions emit interim results that later get corrected;
                                     # file playback always emits True, but Component 2 should not assume that


@dataclass
class Flag:
    """One detected+classified item, emitted by Component 2."""

    type: str            # "question" | "fact"
    text: str            # the flagged span, as spoken
    resolved_text: str   # same, with pronouns/references resolved using context (e.g. "it" -> "the RFP schema")
    speaker: str
    elapsed_seconds: float
    confidence: float
    flag_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    resolved: bool = False
    verdict: Optional[str] = None          # fact: "correct" | "incorrect" | "unverifiable"
    correct_fact: Optional[str] = None     # fact, only when verdict == "incorrect"
    answer: Optional[str] = None           # question
    document_found: Optional[bool] = None  # document_lookup
    citation_document: Optional[str] = None
    citation_url: Optional[str] = None

@dataclass
class RetrievedChunk:
    content: str
    document_name: str
    document_url: Optional[str]
    page_numbers: List[int]
    similarity: float