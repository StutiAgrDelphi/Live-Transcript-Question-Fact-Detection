# component2/test_harness.py
import asyncio
import argparse
from dotenv import load_dotenv
load_dotenv()
from component1.transcript_source import stream_transcript
from component2.pipeline import DetectorPipeline


async def main(path: str, speed: float, window_minutes: float, fire_interval_seconds: float):
    pipeline = DetectorPipeline(
        window_seconds=window_minutes * 60.0,
        fire_interval_seconds=fire_interval_seconds,
    )
    await pipeline.run(stream_transcript(path, playback_speed=speed))
    print(
        f"\nDone. {len(pipeline.dispatcher.questions)} questions, "
        f"{len(pipeline.dispatcher.facts)} facts, "
        f"{len(pipeline.dispatcher.document_lookups)} document lookups flagged."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--speed", type=float, default=20.0)
    p.add_argument("--window-minutes", type=float, default=5.0)
    p.add_argument("--fire-interval-seconds", type=float, default=60.0)
    args = p.parse_args()
    asyncio.run(main(args.path, args.speed, args.window_minutes, args.fire_interval_seconds))