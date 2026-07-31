"""
demo/run_source_demo.py

Run from the project root:

    python -m demo.run_source_demo path/to/transcript.docx --speed 20

--speed 20 plays a ~10 minute meeting back in ~30 seconds — enough to
confirm ordering, pacing, and sentence-splitting are sane without
waiting out the real meeting length.
"""

import argparse
import asyncio

from component1.transcript_source import stream_transcript


async def main(path: str, speed: float):
    print(f"Streaming {path} at {speed}x speed...\n")
    count = 0
    async for chunk in stream_transcript(path, playback_speed=speed):
        count += 1
        print(f"[{chunk.elapsed_seconds:7.1f}s] {chunk.speaker:25s} | {chunk.text}")
    print(f"\nDone — {count} chunks streamed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    asyncio.run(main(args.path, args.speed))