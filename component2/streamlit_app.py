"""
streamlit_app.py

Run with:
    pip install streamlit streamlit-autorefresh
    streamlit run streamlit_app.py

Displays the live transcript (Component 1) alongside the three-way flag
classification (Component 2) as the pipeline runs in a background thread.
"""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import List
import html

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from shared.schema import TranscriptChunk, Flag
from component1.transcript_source import stream_transcript
from component2.pipeline import DetectorPipeline
from component2.dispatcher import Dispatcher
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    transcript: List[TranscriptChunk] = field(default_factory=list)
    questions: List[Flag] = field(default_factory=list)
    facts: List[Flag] = field(default_factory=list)
    document_lookups: List[Flag] = field(default_factory=list)
    done: bool = False
    error: str = ""
    toast_shown: bool = False


class SharedStateDispatcher(Dispatcher):
    """Writes flags into SharedState instead of printing to console."""

    def __init__(self, shared: SharedState):
        self.shared = shared

    def emit(self, flag: Flag):
        with self.shared.lock:
            if flag.type == "question":
                self.shared.questions.append(flag)
            elif flag.type == "document_lookup":
                self.shared.document_lookups.append(flag)
            else:
                self.shared.facts.append(flag)


def _run_pipeline_in_background(
    path: str,
    speed: float,
    window_minutes: float,
    fire_interval_seconds: float,
    shared: SharedState,
):
    async def _main():
        pipeline = DetectorPipeline(
            window_seconds=window_minutes * 60.0,
            fire_interval_seconds=fire_interval_seconds,
        )
        pipeline.dispatcher = SharedStateDispatcher(shared)

        def on_chunk(chunk: TranscriptChunk):
            if not chunk.is_final:
                return
            with shared.lock:
                shared.transcript.append(chunk)

        try:
            await pipeline.run(stream_transcript(path, playback_speed=speed), on_chunk=on_chunk)
        except Exception as e:
            with shared.lock:
                shared.error = str(e)
        finally:
            with shared.lock:
                shared.done = True

    asyncio.run(_main())


def _render_flag_card(flag: Flag, kind: str):
    """Renders a very clean, small-font card for the insights."""
    styles = {
        "question": {"border": "#3b82f6", "icon": "❓"},
        "fact": {"border": "#22c55e", "icon": "📌"},
        "document_lookup": {"border": "#eab308", "icon": "📄"}
    }
    s = styles[kind]
    safe_speaker = html.escape(flag.speaker)
    safe_text = html.escape(flag.resolved_text).replace('\n', '<br>')
    
    card_html = f"""
    <div style="
        border-left: 3px solid {s['border']};
        border-radius: 4px;
        padding: 10px 12px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
        font-family: sans-serif;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; ">
            <span style="font-size: 0.75rem; font-weight: 600; color: #fff;">
                {s['icon']} {safe_speaker}
            </span>
            <span style="font-size: 0.7rem; color: #fff;">
                {flag.elapsed_seconds:.0f}s
            </span>
        </div>
        <div style="font-size: 0.8rem; color: #fff; line-height: 1.4;">
            {safe_text}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def _scroll_to_bottom(marker_id: str):
    st.markdown(f"<div id='{marker_id}'></div>", unsafe_allow_html=True)
    st.components.v1.html(
        f"""
        <script>
        const scrollNow = () => {{
            const target = window.parent.document.getElementById("{marker_id}");
            if (target) target.scrollIntoView({{ block: "end", behavior: "smooth" }});
        }};
        setTimeout(scrollNow, 50);
        setTimeout(scrollNow, 250);
        </script>
        """,
        height=0,
        width=0,
    )

st.markdown(
    """
    <style>
    * {
        scrollbar-width: thin;
        scrollbar-color: rgba(148, 163, 184, 0.75) rgba(15, 23, 42, 0.25);
    }

    *::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    *::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.25);
        border-radius: 999px;
    }

    *::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.75);
        border-radius: 999px;
        border: 2px solid rgba(15, 23, 42, 0.25);
        background-clip: content-box;
    }

    *::-webkit-scrollbar-thumb:hover {
        background: rgba(203, 213, 225, 0.95);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.set_page_config(page_title="Meeting Intelligence", layout="wide", initial_sidebar_state="collapsed")

# --- TOP NAV BAR (Filter & Controls) ---
with st.container():
    st.markdown("<h4 style='margin-bottom: -15px; margin-top:-10px'>Meeting Intelligence</h4>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1], vertical_alignment="bottom")
    
    with c1: path = st.text_input("Transcript file", value="transcript_validation.txt")
    with c2: speed = st.number_input("Speed", value=10.0, min_value=1.0)
    with c3: window_minutes = st.number_input("Window (m)", value=5.0, min_value=1.0)
    with c4: fire_interval = st.number_input("Fire (s)", value=60.0, min_value=10.0)
    with c5: start = st.button("Start Pipeline", use_container_width=True, disabled=st.session_state.get("started", False))

st.divider()

# --- PIPELINE INITIALIZATION ---
if start:
    st.session_state.started = True
    st.session_state.shared = SharedState()
    thread = threading.Thread(
        target=_run_pipeline_in_background,
        args=(path, speed, window_minutes, fire_interval, st.session_state.shared),
        daemon=True,
    )
    thread.start()

if not st.session_state.get("started", False):
    st.info("Configure settings in the top bar and click Start Pipeline.")
    st.stop()

st_autorefresh(interval=1500, key="refresh")

# --- STATE MANAGEMENT ---
shared: SharedState = st.session_state.shared
with shared.lock:
    transcript_snapshot = list(shared.transcript)
    questions_snapshot = list(shared.questions)
    facts_snapshot = list(shared.facts)
    doc_lookups_snapshot = list(shared.document_lookups)
    done = shared.done
    error = shared.error
    
    # Handle the transient success popup safely to avoid hot-reload AttributeErrors
    show_toast = False
    if done and not getattr(shared, "toast_shown", False):
        show_toast = True
        shared.toast_shown = True

if error:
    st.error(f"Pipeline error: {error}")
if show_toast:
    st.toast("Playback finished successfully!", icon="✅")

# --- MAIN LAYOUT ---
# Left: Transcript (approx 35%), Right: Insights (approx 65%)
col_left, col_right = st.columns([3.5, 6.5], gap="large")

# LEFT COLUMN: Transcript
with col_left:
    st.markdown("##### Live Transcript")
    # Vertically tall container
    transcript_box = st.container(height=550)
    with transcript_box:
        for c in transcript_snapshot[-200:]:
            safe_speaker = html.escape(c.speaker)
            safe_text = html.escape(c.text).replace('\n', '<br>')
            st.markdown(f"""
            <div style="margin-bottom: 12px; font-family: sans-serif;">
                <div style="font-size: 0.7rem; color: #64748b; margin-bottom: 2px;">
                    <strong style="color: #334155;">{safe_speaker}</strong> &nbsp;·&nbsp; {c.elapsed_seconds:.0f}s
                </div>
                <div style="border: 1px solid #e2e8f066; padding: 10px 14px; border-radius: 4px 10px 10px 10px; font-size: 0.85rem; color: #fff; line-height: 1.4;">
                    {safe_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        _scroll_to_bottom("transcript-end")
        # # Script to auto-scroll to the bottom of the container
        # st.markdown("<div id='transcript-end'></div>", unsafe_allow_html=True)
        # st.components.v1.html(
        #     """
        #     <script>
        #         setTimeout(() => {
        #             const target = window.parent.document.getElementById('transcript-end');
        #             if(target) {
        #                 target.scrollIntoView({ behavior: 'smooth', block: 'end' });
        #             }
        #         }, 100);
        #     </script>
        #     """,
        #     height=0, width=0
        # )


# RIGHT COLUMN: Insights
with col_right:
    # Top Wide Window: Facts
    st.markdown("##### 📌 Facts")
    fact_box = st.container(height=230)
    with fact_box:
        for flag in reversed(facts_snapshot):
            _render_flag_card(flag, "fact")

        

            
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Bottom Split Windows: Questions & Document Lookups
    col_q, col_d = st.columns(2, gap="large")
    
    with col_q:
        st.markdown("##### ❓ Questions")
        q_box = st.container(height=230)
        with q_box:
            for flag in reversed(questions_snapshot):
                _render_flag_card(flag, "question")

            

        
                
    with col_d:
        st.markdown("##### 📄 Document Lookups")
        d_box = st.container(height=230)
        with d_box:
            for flag in reversed(doc_lookups_snapshot):
                _render_flag_card(flag, "document_lookup")

            

                