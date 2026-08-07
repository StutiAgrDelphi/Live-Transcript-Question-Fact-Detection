# api/server.py
"""
Production WebSocket entrypoint for live meeting transcript ingestion and detection.

Run with:
    pip install fastapi "uvicorn[standard]"
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Two endpoints per live meeting, keyed by a session_id the backend chooses (e.g. the
Teams meeting ID):

  /ws/ingest/{session_id}  — the backend's live-captioning pipeline connects here and
                              sends one JSON message per transcript chunk:
                              {"speaker": "...", "text": "...", "elapsed_seconds": 12.3,
                               "is_final": true}
                              This connection should be opened when the meeting starts
                              and stays open for its duration.

  /ws/flags/{session_id}   — any consumer connects here to receive detected flags for
                              this session as JSON, live, as they're emitted. Requires
                              the ingest connection for the same session_id to already
                              be open (that's what creates the active session) — connect
                              consumers after the ingest side has started.

                              Takes ?user_id=... as a query param. user_ROLE is
                              deliberately NOT accepted from the client — it's looked
                              up server-side from access.users, so a viewer can't just
                              pass ?user_role=admin and grant themselves full access.
"""
import logging
from typing import AsyncIterator, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
load_dotenv()

from shared.schema import TranscriptChunk
from component2.pipeline import DetectorPipeline
from component2.dispatcher import WebSocketDispatcher
from component3.resolving_dispatcher import ResolvingDispatcher
from component3.retrieval import get_user_role

log = logging.getLogger(__name__)
app = FastAPI(title="Meeting Intelligence Detection Service")

from fastapi.staticfiles import StaticFiles
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# session_id -> DetectorPipeline. One active pipeline per live meeting, each with its
# own WindowBuffer/DedupEngine state, so multiple meetings can run detection
# concurrently without interfering with each other.
_active_sessions: Dict[str, DetectorPipeline] = {}


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": list(_active_sessions.keys())}


async def _chunk_source_from_websocket(websocket: WebSocket) -> AsyncIterator[TranscriptChunk]:
    """
    Wraps an inbound WebSocket connection as an AsyncIterator[TranscriptChunk] — the
    same contract component1.transcript_source.stream_transcript already implements.
    This is why DetectorPipeline.run() needs zero changes to work with a live socket.
    """
    while True:
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        try:
            yield TranscriptChunk(
                speaker=data["speaker"],
                text=data["text"],
                elapsed_seconds=float(data["elapsed_seconds"]),
                is_final=bool(data.get("is_final", True)),
            )
        except (KeyError, TypeError, ValueError) as e:
            log.warning(f"Dropping malformed chunk on session, error: {e}")
            continue


@app.websocket("/ws/ingest/{session_id}")
async def ingest_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    pipeline = _active_sessions.get(session_id)
    if pipeline is None:
        pipeline = DetectorPipeline()
        pipeline.dispatcher = ResolvingDispatcher(WebSocketDispatcher())
        _active_sessions[session_id] = pipeline

    try:
        await pipeline.run(_chunk_source_from_websocket(websocket))
    finally:
        # Meeting ended (socket closed) — drop the session so a later meeting
        # reusing this session_id starts with clean buffer/dedup state.
        _active_sessions.pop(session_id, None)


@app.websocket("/ws/flags/{session_id}")
async def flags_endpoint(websocket: WebSocket, session_id: str, user_id: str):
    await websocket.accept()
    pipeline = _active_sessions.get(session_id)
    if pipeline is None or not hasattr(pipeline.dispatcher, "register"):
        await websocket.close(code=4404, reason=f"No active session: {session_id}")
        return

    # Role is NEVER accepted from the client — always looked up server-side.
    user_role = await get_user_role(user_id)

    pipeline.dispatcher.register(websocket, user_id, user_role)
    try:
        while True:
            # This socket is output-only from our side; just keep it open and
            # detect disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        pipeline.dispatcher.unregister(websocket)