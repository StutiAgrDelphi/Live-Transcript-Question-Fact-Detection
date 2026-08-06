"""
component1/parsers.py

Turns a saved transcript file (.txt or .docx export) into a flat,
time-ordered list[TranscriptChunk]. This is the only part of Component 1
that needs to know about file formats — everything downstream just
sees TranscriptChunk.
"""

import re
from datetime import datetime
from typing import List

import docx  # python-docx

from shared.schema import TranscriptChunk


# ---------------------------------------------------------------------------
# .txt format: "[2026-07-14T07:24:05.6649840+00:00] Speaker 1: text..."
# ---------------------------------------------------------------------------

_TXT_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s*(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)


def _parse_txt_timestamp(raw: str) -> datetime:
    # Some exports give 7-digit microseconds, which datetime.fromisoformat
    # rejects (it wants exactly 6). Truncate the fractional part.
    raw = re.sub(r"\.(\d{6})\d+", r".\1", raw)
    return datetime.fromisoformat(raw)


def parse_txt_transcript(path: str) -> List[TranscriptChunk]:
    chunks: List[TranscriptChunk] = []
    first_ts = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _TXT_LINE_RE.match(line)
            if not m:
                continue  # skip anything that doesn't match the expected line shape

            ts = _parse_txt_timestamp(m.group("ts"))
            if first_ts is None:
                first_ts = ts

            chunks.append(
                TranscriptChunk(
                    speaker=m.group("speaker").strip(),
                    text=m.group("text").strip(),
                    elapsed_seconds=(ts - first_ts).total_seconds(),
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# .docx format (Teams meeting recording export): one paragraph per turn,
# shaped like "\n{Speaker Name}   {mm:ss}\n{turn text...}"
# with the speaker name (and only the speaker name) in a bold run.
# ---------------------------------------------------------------------------

_DOCX_TURN_RE = re.compile(
    r"^(?P<speaker>.+?)\s+(?P<ts>\d{1,2}(?::\d{2}){1,2})\s*\n(?P<text>.*)$",
    re.DOTALL,
)


def _parse_docx_timestamp(raw: str) -> float:
    """'3:16' -> 196.0 seconds, '1:02:03' -> 3723.0 seconds."""
    parts = [int(p) for p in raw.split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return float(seconds)


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence split — good enough as a pre-step before the
    LLM does the real semantic work. Keeps '...' and single-word
    fillers ('Yeah.') as their own sentence rather than merging."""
    text = text.replace("\\\n", " ").replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_docx_transcript(path: str, split_turns_into_sentences: bool = True) -> List[TranscriptChunk]:
    doc = docx.Document(path)
    chunks: List[TranscriptChunk] = []

    for para in doc.paragraphs:
        raw = para.text.lstrip("\n")
        m = _DOCX_TURN_RE.match(raw)
        if not m:
            continue  # skip title/date/duration lines and system messages like "started transcription"

        speaker = m.group("speaker").strip()
        turn_start = _parse_docx_timestamp(m.group("ts"))
        turn_text = m.group("text").strip()

        if not split_turns_into_sentences:
            chunks.append(TranscriptChunk(speaker=speaker, text=turn_text, elapsed_seconds=turn_start))
            continue

        # Real live captioning delivers a turn sentence-by-sentence, not as one
        # block — drip these out with small synthetic offsets so playback
        # actually exercises Component 2 the way live delivery would.
        sentences = _split_sentences(turn_text)
        for i, sentence in enumerate(sentences):
            chunks.append(
                TranscriptChunk(
                    speaker=speaker,
                    text=sentence,
                    elapsed_seconds=turn_start + i * 1.5,  # ~1.5s per sentence, arbitrary but monotonic
                )
            )

    return chunks