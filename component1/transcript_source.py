"""
component1/transcript_source.py

Public interface for Component 1. Component 2 (or anyone testing it)
only ever needs this one function — it doesn't know or care that the
data is coming from a file.

    async for chunk in stream_transcript("meeting.docx"):
        ...

Later, when this gets replaced by real speech-to-text, the replacement
just needs to be another `AsyncIterator[TranscriptChunk]` — nothing
downstream should have to change.
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator

from shared.schema import TranscriptChunk
from component1.parsers import parse_txt_transcript, parse_docx_transcript


async def stream_transcript(
    path: str,
    playback_speed: float = 1.0,
) -> AsyncIterator[TranscriptChunk]:
    """Yield TranscriptChunks in order, paced to simulate live delivery.

    playback_speed: 1.0 = real-time gaps between chunks, higher = faster
    (useful while iterating so you're not waiting out a 30-minute meeting
    every test run).
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        chunks = parse_txt_transcript(path)
    elif suffix == ".docx":
        chunks = parse_docx_transcript(path)
    else:
        raise ValueError(f"Unsupported transcript format: {suffix}")

    chunks.sort(key=lambda c: c.elapsed_seconds)

    prev_elapsed = 0.0
    for chunk in chunks:
        gap = (chunk.elapsed_seconds - prev_elapsed) / playback_speed
        if gap > 0:
            await asyncio.sleep(gap)
        prev_elapsed = chunk.elapsed_seconds
        yield chunk