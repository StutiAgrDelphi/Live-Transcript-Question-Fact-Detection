# component2/detector_agent.py
import os
import json
import logging
from typing import List, Dict

from shared.schema import TranscriptChunk, Flag
from agent_framework import create_harness_agent
from agent_framework_openai import OpenAIChatClient

log = logging.getLogger(__name__)

_KB_CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base_context.md")

def _load_kb_context() -> str:
    try:
        with open(_KB_CONTEXT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

SYSTEM_PROMPT = """You are monitoring a live meeting transcript. You are given:
1. CONTEXT — the transcript from roughly the last 5 minutes, for background and
   reference resolution only.
2. NEW SINCE LAST CHECK — the lines spoken since your last check. Only classify
   content from this section.

If the meeting's purpose or agenda has been stated anywhere in CONTEXT or NEW SINCE
LAST CHECK, use it as your relevance yardstick below — whatever domain it turns out
to be. Do not assume a domain in advance.

Go through NEW SINCE LAST CHECK line by line, deciding independently for each line.
A question and its answer spoken separately must each be evaluated on their own.

For each candidate line, work through these checks in order:

1. Grammatical gate (question only): is it interrogative in structure (question mark,
   or question word order — "did you...", "what is...")? An imperative or declarative
   sentence is never a "question," no matter what it requests or proposes.
2. Meeting-mechanics/small-talk check: is this about the people, the room, or the call
   itself, rather than the meeting's actual content — e.g. audio/video/screen-share
   checks, food, weather, sports/personal-life chat, personal favors, or other
   logistics unrelated to why the meeting is happening? If yes, do not flag it —
   regardless of type.
3. Standalone coherence: could you write a complete, understandable resolved_text for
   this using only CONTEXT and the line itself? If it's a disconnected fragment that
   doesn't resolve into a complete thought even with context, skip it.
4. Knowledge-base relevance: if a KNOWLEDGE BASE CONTEXT section appears below, this
   line must relate to the topics, documents, companies, or policies described there —
   either directly referencing that content, asking about it, or making a claim that
   could be verified against it — to qualify for any type. If a KNOWLEDGE BASE CONTEXT
   section is present and this line has nothing to do with any of it, do not flag it,
   even if it would otherwise pass checks 1-3. If no KNOWLEDGE BASE CONTEXT section
   appears below, skip this check (treat it as passed).

Then classify into exactly one type (a line may produce more than one flag if it
genuinely qualifies for more than one):

- "question": passes checks 1-4, AND seeks a decision, clarification, commitment, or
  information affecting the meeting's outcome or someone's action items.
- "fact": passes checks 2-4, AND is a concrete, checkable detail — a specific figure,
  date, decision, or named commitment that others might act on or verify. Not a vague
  opinion or restatement of something already in CONTEXT. A stated intention to
  follow up on something IS a fact (it's a commitment). A sentence with no number,
  date, decision, or named commitment — including a qualitative claim that just
  sounds positive or important, a meta-comment about the meeting itself, or narration
  of what's on screen during a walkthrough — is NOT a fact, even sitting right next to
  one that is.
- "document_lookup": passes checks 2-4, AND is an instruction/request to bring up,
  switch to, or look at a specific named document, section, or topic — or references
  a specific document/company/policy named in the KNOWLEDGE BASE CONTEXT section if
  one is present. Not a vague reference with no identifiable target.

Field guidance:
- "text": the actual words spoken, quoted directly — never a description of what was
  said.
- "resolved_text": the SAME content rewritten as a complete, standalone sentence,
  using CONTEXT to fill in pronouns and missing pieces. If you can't do this
  confidently, don't flag the line — see check 3.
- "passes_relevance_test": a few words on why this line clears checks 1-4 above.

Every line in NEW SINCE LAST CHECK is labeled "[speaker @ Xs]". Copy the exact speaker
name and numeric X value from that label into "speaker" and "elapsed_seconds" — do not
invent or approximate these.

Return ONLY valid JSON, no preamble, no markdown fences:
{"flags": [{"type": "question"|"fact"|"document_lookup", "text": "...",
"resolved_text": "...", "speaker": "...", "elapsed_seconds": 0.0, "confidence": 0.0-1.0,
"passes_relevance_test": "..."}]}
If nothing qualifies, return {"flags": []}."""


def _format_lines(chunks: List[TranscriptChunk]) -> str:
    return "\n".join(f"[{c.speaker} @ {c.elapsed_seconds:.1f}s] {c.text}" for c in chunks)


def to_flag(raw: dict, fallback_chunk: TranscriptChunk) -> Flag:
    try:
        elapsed = float(raw.get("elapsed_seconds", fallback_chunk.elapsed_seconds))
    except (TypeError, ValueError):
        elapsed = fallback_chunk.elapsed_seconds
    return Flag(
        type=raw.get("type", "fact"),
        text=raw.get("text", ""),
        resolved_text=raw.get("resolved_text", ""),
        speaker=raw.get("speaker") or fallback_chunk.speaker,
        elapsed_seconds=elapsed,
        confidence=float(raw.get("confidence", 1.0)),
    )


class DetectorAgent:
    def __init__(self):
        client = OpenAIChatClient(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="preview",
        )

        instructions = SYSTEM_PROMPT
        kb_context = _load_kb_context()
        if kb_context:
            instructions += f"\n\nKNOWLEDGE BASE CONTEXT:\n{kb_context}"

        self.agent = create_harness_agent(
            client=client,
            agent_instructions=instructions,
            name="TranscriptDetector",
            description="Detects relevant questions and facts from live meeting transcripts.",
            disable_tool_auto_approval=True,
            
        )

    async def process_window(
        self,
        context_chunks: List[TranscriptChunk],
        new_chunks: List[TranscriptChunk],
    ) -> List[Dict]:
        context_str = _format_lines(context_chunks)
        new_str = _format_lines(new_chunks)
        prompt = f"CONTEXT:\n{context_str}\n\nNEW SINCE LAST CHECK:\n{new_str}"
        try:
            session = self.agent.create_session()
            response = await self.agent.run(
                prompt,
                session=session,
            )
            output = response.text.strip()

            # defensive JSON parsing (strip fences if present)
            if output.startswith("```"):
                output = "\n".join(output.split("\n")[1:-1]).strip()
            if output.startswith("json"):
                output = output[4:].strip()

            data = json.loads(output)
            return data.get("flags", [])
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON from agent response: {e}")
            return []
        except Exception as e:
            log.warning(f"Error during agent processing: {e}")
            return []