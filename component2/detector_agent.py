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
2. Meeting-mechanics/small-talk check: is this about the people, the room, or the
   call itself, rather than the meeting's actual content — e.g. audio/video/screen-
   share checks, food, weather, sports/personal-life chat, personal favors,
   someone stepping away or being interrupted, pauses in discussion, or other
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

- "question": passes checks 1-4, AND is genuinely interrogative in the ORIGINAL
  line (see check 1), AND seeks a decision, clarification, commitment, or
  information that isn't already settled — something someone in the meeting is
  actually uncertain about, or wants confirmed or looked up.

  Calibration:
    A tag-question or rhetorical check that gets answered later in the SAME
    NEW SINCE LAST CHECK batch is STILL a "question" at the moment it's asked —
    flag it as a question AND flag the answer separately as its own fact. Don't
    collapse the two into a single fact and silently drop the question.
      e.g. "Was it $115.2 billion? ... No, that's the full year figure, Q4 was
      $35.6 billion" → TWO flags: a question ("Is the figure $115.2 billion?")
      and a fact ("Q4 revenue was $35.6 billion") — never just the fact alone.
    A rhetorical question that only introduces a statement the speaker is about
    to make themselves, with no real uncertainty (e.g. "You know what stood out
    to me? The numbers were huge.") is NOT a "question" — evaluate the statement
    that follows it as the candidate line instead.

  Consistency rule: "type" and "resolved_text" must always agree. If
  resolved_text is phrased as an interrogative ("Does X confirm Y?", "Is Z
  true?"), type MUST be "question." If type is "fact," resolved_text MUST be
  declarative. Never emit an interrogative resolved_text under type "fact," or
  a declarative resolved_text under type "question."
- "fact": passes checks 2-4, AND is a concrete, checkable claim about what a
  document or topic actually SAYS, STATES, DEFINES, or REQUIRES — something
  that could plausibly appear as a sentence inside the source document itself.

  A fact is NEVER:
    - An action item or task assignment — who will do, review, verify, or
      follow up on something. This is true even if it names a real document
      from the knowledge base ("Anna will review the SBI policy" is about
      Anna, not about what the SBI policy says — do not flag it, and it must
      never be checked against the knowledge base).
    - A meta-comment about the meeting itself (a question "has been
      answered," a topic "being discussed," someone "reading" or "looking
      at" a document).
    - Someone's personal opinion or reaction attributed to them ("Alok found
      the definitions precise," "the team thinks this is the hardest
      document") — this is a claim about what a person believes, not about
      the document, and checking it against the document will always
      produce a meaningless verdict either way.
    - Meeting mechanics or someone's personal circumstance, even when it
      happens right next to KB-relevant discussion (stepping away from the
      call, being interrupted, a pause in discussion, "I read this
      yesterday"). These belong under check 2 — treat them as excluded there
      regardless of what's being discussed around them.

  Test before flagging — apply BOTH tests:
  1. Document-fit test: if you dropped this sentence into the actual source
     document, would it fit as something the document itself might say?
     If it describes a person's action, opinion, presence, or the meeting
     itself instead, it fails — do not flag it.
  2. Interpretation-verb test: does the sentence use a verb like "presents,"
     "frames," "emphasizes," "conveys," "positions," "pushes," "describes,"
     "treats," "indicates," or "highlights"? These are interpretation verbs
     — they describe what a reader thinks about a document, not what the
     document actually states. A sentence with these verbs is commentary, not
     a fact — discard it, even if it names a real document and a real number.
     Exception: if the verb is "states," "says," "defines," "requires,"
     "reports," or "confirms," that's a direct-quote verb and the sentence
     can proceed to the normal fact test.

  NOT a fact (generic examples — same reasoning applies regardless of domain):
    "Maria will review the compliance policy before Friday." (action item)
    "James thinks the contract is hard to read." (opinion about a document)
    "That question has already been answered." (meta-comment about meeting)
    "Priya was reading the report yesterday." (personal activity)
    "We paused the discussion for a moment." (meeting mechanics)
      - A characterization or interpretation of what a document "presents,"
      "frames," "emphasizes," "describes," "conveys," or "positions" — even
      if true. ("The report frames NVIDIA as an AI factory" is NOT a fact
      you can check against the report; only "The report states X" where X
      is a specific verbatim or near-verbatim claim is checkable.) If the
      verb is interpret/frame/emphasize/convey/position/push, discard it.
    - A comparison between two documents or a general observation about
      document style ("Document A is expansive, Document B is specific,"
      "this is a definition-heavy policy," "policy documents usually focus
      on key clauses") — these are observations about documents, not claims
      from them.
    - A recap or summary of multiple facts already stated earlier in the
      CONTEXT window. If the line is clearly a wrap-up of what was just
      discussed (often starting with "so," "in summary," "to recap," "the
      key facts are"), treat it as a meta-comment — do not flag it as a
      fact, even if it lists specific numbers, because a downstream
      fact-checker will retrieve different passages for each number and get
      confused by the mixed claim.

  IS a fact:
    "The compliance policy requires annual review by December 31st."
    "The contract states termination requires 30 days written notice."
  
  The difference is never about which document or topic is being discussed — it's
  whether the sentence itself carries a number, date, named decision, or named
  commitment. Before flagging, apply this test: if you removed every number, date,
  and proper noun from the line, would any checkable claim remain? If no, it's
  commentary — discard it, even if it's clearly about something in the knowledge
  base and even if it "sounds important."
  
  If a NEW line contains more than one independently checkable clause (e.g. two
  "if X then Y" statements joined by a period, or a list of separate commitments),
  evaluate and flag EACH clause on its own. Do not skip a later clause just because
  an earlier one in the same line already produced a flag — repetition of sentence
  shape is not repetition of content.
- "document_lookup": passes checks 2-4, AND the conversation's focus is shifting to
  (or opening on) a specific named document, section, or topic described in the
  KNOWLEDGE BASE CONTEXT section, if one is present. This includes:
    - Explicit retrieval requests ("show me...", "pull up...", "find the section on...").
    - Natural spoken topic transitions that name or clearly identify a DIFFERENT
      document than what's currently in focus — e.g. "let's start with...",
      "now let's move to...", "coming back to...", "switching to...". These aren't
      phrased as commands, but they still mean a different document needs to come
      into view, which is exactly what this flag exists to catch.
    - The opening line of a meeting, if it names or clearly identifies which
      document the discussion is beginning with.
  Only flag the line that INTRODUCES the shift. Once a document is the active
  topic, further discussion of it is fact/question territory — do not re-flag
  document_lookup just because the document's name comes up again.
  NOT: a request to focus on a sub-topic WITHIN the document already active
  (e.g. "let's look at the arbitration clause" when that clause, inside the
  document already in focus, is what's already being discussed). That's staying
  inside the same document, not shifting to a new one — do not flag it.
  When the document isn't named explicitly in the line itself (e.g. "let's start
  with the annual report," no company given), use the rest of NEW SINCE LAST
  CHECK — not just CONTEXT — to identify which specific document from KNOWLEDGE
  BASE CONTEXT is meant, the same way you would for a fact.
  
  If the line is clearly a request to look at, pull up, or open SOME document or
  file (e.g. "let's look into that file," "can you bring that up"), but no
  specific document can be confidently identified from CONTEXT or the rest of
  NEW SINCE LAST CHECK, still flag it as "document_lookup" — do not skip it, and
  do not guess a title. In this case, resolved_text must stay close to what was
  actually said, generically (e.g. "Look into the file being referenced" or a
  close rephrasing) — never invent a specific document name, title, or KB entry
  that wasn't clearly identifiable. Only name a specific document when you can
  confidently resolve one from CONTEXT or NEW SINCE LAST CHECK — never as a
  placeholder guess.
  
  resolved_text must always be a retrieval instruction, never a description of
  the line itself or of your own detection process (e.g. never "the document
  being discussed at the start of the meeting" — that describes what you
  noticed, not what to retrieve).
  
  If the line doesn't even clearly express an intent to look at some document
  (too vague or ambiguous to tell), don't flag it — see check 3.

Field guidance:
- "text": the actual words spoken, quoted directly — never a description of what was
  said.
- "resolved_text": a rewritten, standalone version optimized for a downstream RAG
  sub-agent to search the knowledge base with — NOT a lightly cleaned copy of the
  spoken line. Use CONTEXT to fill in pronouns and missing pieces, and:
    - For "fact": phrase it as a checkable claim, naming the document it relates
      to using its EXACT title from KNOWLEDGE BASE CONTEXT (e.g. "Nvidia 2025
      Annual Report", not "the annual report"). If KNOWLEDGE BASE CONTEXT lists
      more than one document of the same general kind, a generic label like "the
      annual report" or "the policy" is NOT specific enough — it's unusable for
      retrieval. Use CONTEXT *and the rest of NEW SINCE LAST CHECK* (the whole
      batch, not just this line) to work out which exact one is meant, by
      matching entities, numbers, or topics against each document's description.
      Only use a generic label if the knowledge base genuinely contains just one
      document of that kind. If you truly cannot resolve which specific document
      is meant even using the full batch, state the claim without a document
      name rather than guessing wrong.
    - For "question": phrase it as an explicit verification query against the
      knowledge base, naming the document using its EXACT title from KNOWLEDGE BASE
      CONTEXT (e.g. "Does the Nvidia 2025 Annual Report confirm revenue of
      $130.5 billion?"). If KNOWLEDGE BASE CONTEXT contains more than one document
      of the same kind (for example, multiple annual reports or multiple policies),
      never use a generic name like "the annual report" or "the policy". Use
      CONTEXT and the rest of NEW SINCE LAST CHECK to determine which exact
      document is being referred to. If you genuinely cannot determine the correct
      document, ask the question without naming a document rather than guessing.
    - For "document_lookup": phrase it as a direct retrieval instruction naming
      the specific document (e.g. "Retrieve the SBI Health Policy document"
      rather than "Now let's switch to the SBI health policy").
  If you cannot produce a confident, complete resolved_text this way, don't flag
  the line — see check 3.
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
            default_options={"temperature": 0.1}
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