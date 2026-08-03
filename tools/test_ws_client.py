# tools/test_ws_client.py
"""
Local test client for the WebSocket API layer (api/server.py). Does two things:
  1. Connects to /ws/ingest/{session} and streams a transcript file into it —
     same source (component1.transcript_source.stream_transcript) your file-based
     tests already use, just sent over a real socket instead of called in-process.
  2. Connects to /ws/flags/{session} and prints every detected flag as it arrives.

Start the server first, in a separate terminal, from the project root:
    uvicorn api.server:app --reload

Then run this, from the project root, in another terminal:
    python -m tools.test_ws_client transcript_validation.txt --speed 20 --session test1
"""
import asyncio
import argparse
import json
from dataclasses import asdict

import websockets

from component1.transcript_source import stream_transcript

_TAB_NAMES = {"question": "QUESTIONS", "fact": "FACTS", "document_lookup": "DOCUMENT LOOKUP"}


async def _send_transcript(uri: str, path: str, speed: float):
    async with websockets.connect(uri) as ws:
        async for chunk in stream_transcript(path, playback_speed=speed):
            await ws.send(json.dumps(asdict(chunk)))
    print("Finished sending transcript.")


async def _print_flags(uri: str):
    async with websockets.connect(uri) as ws:
        async for message in ws:
            flag = json.loads(message)
            tab = _TAB_NAMES.get(flag["type"], "FACTS")
            print(f"[{tab}] ({flag['speaker']} @ {flag['elapsed_seconds']:.1f}s) {flag['resolved_text']}")


async def main(path: str, speed: float, session: str, host: str):
    ingest_uri = f"ws://{host}/ws/ingest/{session}"
    flags_uri = f"ws://{host}/ws/flags/{session}"

    sender = asyncio.create_task(_send_transcript(ingest_uri, path, speed))
    await asyncio.sleep(1.0)  # let the ingest side create the session first
    listener = asyncio.create_task(_print_flags(flags_uri))

    await sender
    await asyncio.sleep(5.0)  # let any in-flight flags finish arriving
    listener.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--speed", type=float, default=15.0)
    p.add_argument("--session", default="test-session")
    p.add_argument("--host", default="localhost:8000")
    args = p.parse_args()
    asyncio.run(main(args.path, args.speed, args.session, args.host))