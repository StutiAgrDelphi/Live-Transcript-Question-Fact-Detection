# component2/pipeline.py
import logging
from typing import AsyncIterator, Callable, Optional
from shared.schema import TranscriptChunk
from component2.buffer import WindowBuffer
from component2.dedup import DedupEngine
from component2.detector_agent import DetectorAgent, to_flag
from component2.dispatcher import ConsoleDispatcher

log = logging.getLogger(__name__)


class DetectorPipeline:
    def __init__(
        self,
        window_seconds: float = 300.0,
        fire_interval_seconds: float = 90.0,
    ):
        self.buffer = WindowBuffer(window_seconds=window_seconds)
        self.dedup = DedupEngine()
        self.agent = DetectorAgent()
        self.dispatcher = ConsoleDispatcher()
        self.fire_interval_seconds = fire_interval_seconds
        self._next_fire_at = fire_interval_seconds

    async def run(
        self,
        source: AsyncIterator[TranscriptChunk],
        on_chunk: Optional[Callable[[TranscriptChunk], None]] = None,
    ):
        async for chunk in source:
            try:
                if on_chunk:
                    on_chunk(chunk)
                if not chunk.is_final:
                    continue
                self.buffer.add(chunk)
                if chunk.elapsed_seconds >= self._next_fire_at:
                    await self._fire()
                    self._next_fire_at += self.fire_interval_seconds
            except Exception as e:
                log.warning(f"Skipping chunk due to error: {e}")
                continue
        await self._fire()

    async def _fire(self):
        context_chunks, new_chunks = self.buffer.get_window_and_new()
        if not new_chunks:
            return
        try:
            raw_flags = await self.agent.process_window(context_chunks, new_chunks)
        except Exception as e:
            log.warning(f"Skipping window fire due to error: {e}")
            return
        fallback_chunk = new_chunks[-1]
        for rf in raw_flags:
            flag = to_flag(rf, fallback_chunk)
            if not self.dedup.is_duplicate(flag):
                self.dispatcher.emit(flag)
        self.buffer.mark_flushed(new_chunks[-1].elapsed_seconds)