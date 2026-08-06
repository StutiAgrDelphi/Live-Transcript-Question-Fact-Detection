# component2/buffer.py
from collections import deque
from typing import List, Tuple
from shared.schema import TranscriptChunk
from component2.segmenter import split_sentences


class WindowBuffer:
    """
    Accumulates whole, finalized TranscriptChunks. Exposes:
      - the full rolling context window (last `window_seconds` of meeting time)
      - the subset of chunks that are new since the last classification firing

    Chunks are never split — always handled as the whole atomic unit Component 1
    already produced (a full sentence for docx-sourced chunks, a full line for
    txt-sourced chunks).
    """

    def __init__(self, window_seconds: float = 300.0):
        self.window_seconds = window_seconds
        self.chunks: deque[TranscriptChunk] = deque()
        self._last_flushed_elapsed: float = -1.0

    # def add(self, chunk: TranscriptChunk):
    #     self.chunks.append(chunk)
    #     self._prune(chunk.elapsed_seconds)

    def add(self, chunk: TranscriptChunk):
        sentences = split_sentences(chunk.text)
        if len(sentences) <= 1:
            self.chunks.append(chunk)
        else:
            for i, sentence in enumerate(sentences):
                self.chunks.append(TranscriptChunk(
                    speaker=chunk.speaker,
                    text=sentence,
                    elapsed_seconds=chunk.elapsed_seconds + i * 0.1,
                    is_final=chunk.is_final,
                ))
        self._prune(chunk.elapsed_seconds)

    def _prune(self, now_elapsed: float):
        cutoff = now_elapsed - self.window_seconds
        while self.chunks and self.chunks[0].elapsed_seconds < cutoff:
            self.chunks.popleft()

    def get_window_and_new(self) -> Tuple[List[TranscriptChunk], List[TranscriptChunk]]:
        """Returns (full_window, new_since_last_fire). Both preserve arrival order."""
        window = list(self.chunks)
        new = [c for c in window if c.elapsed_seconds > self._last_flushed_elapsed]
        return window, new

    def mark_flushed(self, up_to_elapsed: float):
        self._last_flushed_elapsed = up_to_elapsed