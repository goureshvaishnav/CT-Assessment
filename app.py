"""
app.py — ML Lab: RAG Chatbot
Streamlit UI for a local LLM + ChromaDB document chatbot.
"""

import os
import tempfile
import time

import streamlit as st
from rag_engine import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_HISTORY,
    TOP_K_DOCS,
    RAGEngine,
)

# ──────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot · ML Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ─────────────────────────── */
:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #1f2535;
    --accent:    #4fffb0;
    --accent2:   #3d8eff;
    --danger:    #ff4e6a;
    --text:      #e4e8f5;
    --muted:     #6b7496;
    --code-bg:   #1a1e2e;
}

/* ── Global ───────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* ── Hide Streamlit chrome ────────────────── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }

/* Keep chat area comfortably readable on large screens */
[data-testid="stMainBlockContainer"] {
    max-width: 980px !important;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Header strip ─────────────────────────── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
    background: linear-gradient(
        120deg,
        rgba(79,255,176,.10) 0%,
        rgba(61,142,255,.06) 40%,
        transparent 85%
    );
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, .25);
}
.rag-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent) !important;
    margin: 0;
    letter-spacing: -.5px;
}
.rag-header p {
    font-size: .78rem;
    color: var(--muted);
    margin: 2px 0 0;
}
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: .68rem;
    font-family: 'Space Mono', monospace;
    background: rgba(79,255,176,.12);
    color: var(--accent);
    border: 1px solid rgba(79,255,176,.25);
    margin: 2px 3px;
}
.badge.blue { background: rgba(61,142,255,.12); color: var(--accent2); border-color: rgba(61,142,255,.25); }

/* ── Status bar ───────────────────────────── */
.status-bar {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    padding: 10px 0 16px;
}
.stat-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 24px;
    font-size: .75rem;
    font-family: 'Space Mono', monospace;
    border: 1px solid var(--border);
    background: var(--surface);
}
.stat-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.stat-pill.online .dot { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
.stat-pill.offline .dot { background: var(--danger); }
.stat-pill.docs    .dot { background: var(--accent2); box-shadow: 0 0 6px var(--accent2); }

/* ── Chat messages ────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    margin-bottom: .8rem !important;
    padding: 16px 18px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .18);
}
[data-testid="stChatMessage"][data-role="user"] {
    border-color: rgba(61,142,255,.3) !important;
    background: linear-gradient(180deg, rgba(61,142,255,.08), rgba(61,142,255,.03)) !important;
}
[data-testid="stChatMessage"][data-role="assistant"] {
    border-color: rgba(79,255,176,.2) !important;
    background: linear-gradient(180deg, rgba(79,255,176,.06), rgba(79,255,176,.02)) !important;
}

/* ── Chat input ───────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid rgba(79,255,176,.3) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .9rem !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, .20) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,255,176,.15) !important;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .82rem !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(79,255,176,.07) !important;
}

/* ── File uploader ────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
}

/* ── Expander ─────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: .85rem !important;
}

/* ── Code blocks ──────────────────────────── */
code, pre {
    background: var(--code-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── Section label ────────────────────────── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: .65rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1rem 0 .5rem;
}

/* ── Empty state ──────────────────────────── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
}
.empty-state .icon { font-size: 2.8rem; margin-bottom: 12px; }
.empty-state h3 { font-size: 1rem; color: var(--text); margin-bottom: 8px; }
.empty-state p  { font-size: .82rem; line-height: 1.6; }

/* ── Divider ──────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
def _init_state():
    if "rag" not in st.session_state:
        st.session_state.rag = RAGEngine()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

_init_state()
rag: RAGEngine = st.session_state.rag


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
    <div style="padding:12px 0 20px">
        <div style="font-family:'Space Mono',monospace;font-size:.7rem;
                    color:#4fffb0;letter-spacing:.15em;text-transform:uppercase;
                    margin-bottom:4px">ML Lab Project</div>
        <div style="font-size:1.1rem;font-weight:600">RAG Chatbot</div>
        <div style="font-size:.72rem;color:#6b7496;margin-top:4px">
            Local LLM · ChromaDB · LangChain
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── System status ─────────────────────────
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    health = rag.get_health_report()
    ollama_ok = health["ollama"]
    embed_ok = health["embed_model"]
    llm_ok = health["llm_model"]
    doc_count = health["doc_chunks"]

    ollama_class  = "online" if ollama_ok     else "offline"
    embed_class   = "online" if embed_ok      else "offline"
    llm_class     = "online" if llm_ok        else "offline"
    docs_class    = "docs"   if doc_count > 0 else ""

    st.markdown(f"""
    <div class="status-bar">
        <div class="stat-pill {ollama_class}">
            <div class="dot"></div>
            Ollama {"running" if ollama_ok else "offline"}
        </div>
        <div class="stat-pill {embed_class}">
            <div class="dot"></div>
            {EMBED_MODEL} {"ready" if embed_ok else "missing"}
        </div>
        <div class="stat-pill {llm_class}">
            <div class="dot"></div>
            {LLM_MODEL} {"ready" if llm_ok else "missing"}
        </div>
        <div class="stat-pill {docs_class}">
            <div class="dot"></div>
            {doc_count} chunks
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not ollama_ok:
        st.warning("Run `ollama serve` to start the LLM server.", icon="⚡")
    if ollama_ok and not llm_ok:
        st.info(f"Pull the LLM model:\n```\nollama pull {LLM_MODEL}\n```", icon="📥")
    if ollama_ok and not embed_ok:
        st.info(f"Pull the embed model:\n```\nollama pull {EMBED_MODEL}\n```", icon="📥")

    st.markdown('<div class="section-label">First Run Checklist</div>', unsafe_allow_html=True)
    st.checkbox("Ollama server is running", value=ollama_ok, disabled=True)
    st.checkbox(f"Embedding model `{EMBED_MODEL}` available", value=embed_ok, disabled=True)
    st.checkbox(f"LLM model `{LLM_MODEL}` available", value=llm_ok, disabled=True)
    st.checkbox("At least one document chunk indexed", value=doc_count > 0, disabled=True)

    st.divider()

    # ── Upload PDFs ───────────────────────────
    st.markdown('<div class="section-label">Document Ingestion</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF files to add to the knowledge base.",
    )

    if uploaded_files:
        tmp_dir   = tempfile.mkdtemp()
        new_paths = []

        for f in uploaded_files:
            path = os.path.join(tmp_dir, f.name)
            with open(path, "wb") as fh:
                fh.write(f.getvalue())
            new_paths.append(path)

        if new_paths:
            with st.spinner(f"Indexing {len(new_paths)} file(s)…"):
                result = rag.load_pdfs(new_paths)

            if result["files_indexed"] > 0:
                st.success(
                    f"Indexed {result['files_indexed']} file(s) into {result['chunks']} chunks",
                    icon="✅",
                )
                st.session_state.indexed_files = sorted(list(
                    set(st.session_state.indexed_files + result["indexed_files"])
                ))
            if result["files_skipped"] > 0:
                skipped_preview = ", ".join(result["skipped_files"][:3])
                suffix = "..." if len(result["skipped_files"]) > 3 else ""
                st.info(
                    f"Skipped {result['files_skipped']} duplicate file(s): "
                    f"{skipped_preview}{suffix}",
                    icon="ℹ️",
                )
            if result["errors"]:
                for err in result["errors"]:
                    st.error(f"❌ {err['file']}: {err['error']}", icon="❌")

    # Indexed files list
    if st.session_state.indexed_files:
        with st.expander(f"📚 Indexed files ({len(st.session_state.indexed_files)})", expanded=False):
            for fname in st.session_state.indexed_files:
                st.markdown(
                    f'<div style="font-size:.75rem;padding:3px 0;color:#a0aac4">📄 {fname}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Controls ──────────────────────────────
    st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Reset DB", use_container_width=True, help="Wipe ChromaDB and start fresh"):
            with st.spinner("Resetting…"):
                rag.reset_vectorstore()
                st.session_state.messages       = []
                st.session_state.indexed_files = []
                st.session_state.last_sources = []
            st.success("Reset complete!")
            time.sleep(1)
            st.rerun()
    with col2:
        if st.button("🧹 Clear chat", use_container_width=True, help="Clear conversation history only"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # ── Model info ────────────────────────────
    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:.75rem;line-height:1.9;color:#a0aac4">
        🤖 <b>LLM</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {LLM_MODEL}<br>
        🔢 <b>Embed</b> &nbsp;&nbsp; {EMBED_MODEL}<br>
        💾 <b>Vector DB</b> ChromaDB (local)<br>
        🔗 <b>Framework</b> LangChain<br>
        🧩 <b>Chunking</b> {CHUNK_SIZE}/{CHUNK_OVERLAP} (size/overlap)<br>
        🎯 <b>Retrieval</b> top-{TOP_K_DOCS} · history {MAX_HISTORY} turns
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main — Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div style="font-size:2rem">🧠</div>
    <div>
        <h1>RAG Chatbot</h1>
        <p>AI assistant trained on your documents — fully local, zero cloud cost</p>
    </div>
    <div style="margin-left:auto;display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end">
        <span class="badge">Mistral 7B</span>
        <span class="badge blue">ChromaDB</span>
        <span class="badge">LangChain</span>
        <span class="badge blue">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Main — Chat area
# ──────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info(
            "If you do not see the left sidebar, click the small arrow at the "
            "top-left to expand it and upload PDFs.",
            icon="⬅️",
        )
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📂</div>
            <h3>Upload documents to get started</h3>
            <p>
                Drop one or more PDFs in the sidebar.<br>
                Once indexed, ask any question about your documents<br>
                and the assistant will answer with cited context.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
            if msg.get("role") == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                    for idx, source in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**[{idx}]** `{source['file']}` · page "
                            f"`{source['page']}` · score `{source['score']}`"
                        )
                        st.caption(source["snippet"])

# ──────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            result = rag.query_with_sources(
                question=prompt,
                chat_history=st.session_state.messages[:-1],   # exclude current prompt
            )
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})", expanded=False):
                for idx, source in enumerate(result["sources"], start=1):
                    st.markdown(
                        f"**[{idx}]** `{source['file']}` · page "
                        f"`{source['page']}` · score `{source['score']}`"
                    )
                    st.caption(source["snippet"])
        st.session_state.last_sources = result["sources"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
