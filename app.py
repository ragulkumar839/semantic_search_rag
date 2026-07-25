import os
import re
import time
import streamlit as st

from sentence_transformers import SentenceTransformer

from src.pdf_loader import load_pdf
from src.chunking import chunk_text
from src.embedding import create_embeddings
from src.vector_db import (
    create_index,
    save_index,
    load_index
)
from src.bm25_search import build_bm25
from src.hybrid_search import hybrid_search
from src.rag import generate_answer, FALLBACK_MARKER, get_key_diagnostic, get_last_model_errors


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="AI Document Assistant Pro | Enterprise RAG",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# Custom Professional CSS (Dark Glassmorphism Theme & No White Lines)
# -------------------------

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Force dark theme variables across Streamlit root */
:root {
    --background-color: #0f172a !important;
    --secondary-background-color: #1e293b !important;
    --text-color: #f8fafc !important;
}

/* App Background: Dark Obsidian Radial Gradient */
.stApp,
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewMain"] {
    background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #0f172a 50%, #090d16 100%) !important;
    color: #f8fafc !important;
}

/* Header transparency */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85) !important;
    backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* FIX STREAMLIT BOTTOM CHAT INPUT WHITE BANNER */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
footer,
footer * {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stBottom"]::before,
[data-testid="stBottom"]::after,
div[data-testid="stBottom"]::before,
div[data-testid="stBottom"]::after {
    background: transparent !important;
    display: none !important;
}

/* FIX STREAMLIT EXPANDER WHITE HEADER & BOX */
div[data-testid="stExpander"],
details[data-testid="stExpander"],
summary[data-testid="stExpanderSummary"],
summary[data-testid="stExpanderSummary"] *,
.streamlit-expanderHeader,
.streamlit-expanderHeader *,
div[data-testid="stExpanderDetails"],
div[data-testid="stExpanderDetails"] * {
    background-color: #1e293b !important;
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
}

summary[data-testid="stExpanderSummary"]:hover,
.streamlit-expanderHeader:hover {
    background-color: #334155 !important;
    background: #334155 !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
}

/* FIX BASEWEB INPUT CONTAINERS & CHAT INPUT WRAPPER WHITE BOX */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-testid="stChatInputContainer"],
form[data-testid="stChatInputForm"] {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #ffffff !important;
    border-radius: 9999px !important;
}

/* REMOVE ALL WHITE LINES / HARSH DIVIDERS */
hr, [data-testid="stDivider"], div[data-testid="stDivider"], hr[data-testid="stDivider"] {
    border: none !important;
    border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
    background: transparent !important;
    margin: 1.2rem 0 !important;
}

section[data-testid="stSidebar"] hr {
    border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
    margin: 0.8rem 0 !important;
}

/* Hero Title Header */
.hero-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc 0%, #6366f1 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 500;
    max-width: 700px;
    margin: 0 auto 1rem auto;
    line-height: 1.5;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
    margin: 3px;
}

.status-online {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

/* Chat Message Container - Dark Sleek Border */
div[data-testid="stChatMessage"] {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 18px !important;
    padding: 1.25rem !important;
    margin-bottom: 1.2rem !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3) !important;
    color: #ffffff !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div,
div[data-testid="stChatMessage"] li {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    font-size: 1.02rem !important;
    line-height: 1.6 !important;
}

div[data-testid="stChatMessage"]:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
}

/* Distinct User Message Styling */
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
    background: #312e81 !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
}

div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) p,
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 600 !important;
}

/* Floating Chat Input Pill */
div[data-testid="stChatInput"] {
    border-radius: 9999px !important;
    border: 1.5px solid rgba(129, 140, 248, 0.5) !important;
    background-color: #0f172a !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
}

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input,
div[data-testid="stChatInput"] p,
div[data-testid="stChatInput"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    caret-color: #38bdf8 !important;
}

div[data-testid="stChatInput"] textarea::placeholder,
div[data-testid="stChatInput"] input::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

/* General Input Contrast */
input, textarea, .stTextInput input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background-color: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Primary Action Buttons */
.stButton button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6) !important;
}

/* Secondary Download Buttons */
.stDownloadButton button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #e2e8f0 !important;
}

.stDownloadButton button:hover {
    background: rgba(51, 65, 85, 1) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
}

/* Sidebar Metric Cards */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
}

/* Source Expander Card */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* Citation Badges */
.badge-high {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.badge-medium {
    background: rgba(234, 179, 8, 0.15);
    color: #fde047;
    border: 1px solid rgba(234, 179, 8, 0.3);
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.badge-low {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
}

/* FIX WHITE <code> PILL BEHIND CITATION SOURCE FILENAMES */
code,
p code,
span code,
div code,
.stMarkdown code,
div[data-testid="stChatMessage"] code,
div[data-testid="stMarkdownContainer"] code {
    background-color: rgba(99, 102, 241, 0.15) !important;
    color: #a5b4fc !important;
    -webkit-text-fill-color: #a5b4fc !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.9rem !important;
}

/* Welcome Card */
.welcome-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px dashed rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------
# Startup Credential Check
# -------------------------

def _has_api_key():
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return True
    try:
        return bool(st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        return False

if not _has_api_key():
    st.error(
        "⚠️ No Gemini API key found. Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) as an "
        "environment variable / `.env` entry for local runs, or under **Settings > Secrets** "
        "if deployed on Streamlit Community Cloud — answer generation will fail without it."
    )
    st.stop()

# -------------------------
# Helpers
# -------------------------

def sanitize_filename(name: str) -> str:
    """Strip any path components and unsafe characters from an uploaded filename."""
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "unnamed.pdf"

def get_answer(query, results):
    """generate_answer() already retries per-model and fails over across the
    candidate model list internally (see src/rag.py), returning a plain-text
    extractive fallback — prefixed with FALLBACK_MARKER — if every model is
    unavailable. This wrapper only needs to catch truly unexpected errors
    (e.g. a bug before the model loop even runs) and flag degraded mode.
    Returns (answer_text, degraded: bool)."""
    try:
        answer = generate_answer(query, results)
    except Exception as e:
        return f"⚠️ Error generating answer: {e}", True

    return answer, answer.startswith(FALLBACK_MARKER)

# -------------------------
# Load Embedding Model
# -------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# -------------------------
# Initialize Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -------------------------
# Automatically Load Existing Knowledge Base
# -------------------------

kb_ready = False
if "index" not in st.session_state:
    index, chunks = load_index()
    if index is not None:
        st.session_state["index"] = index
        st.session_state["chunks"] = chunks
        build_bm25(chunks)
        kb_ready = True

if "index" in st.session_state and st.session_state.get("chunks"):
    kb_ready = True

# -------------------------
# Hero Banner & Status Pill
# -------------------------

status_pill = (
    f'<span class="badge-pill status-online">🟢 Knowledge Base Ready ({len(st.session_state.get("chunks", []))} Chunks)</span>'
    if kb_ready
    else '<span class="badge-pill" style="background:rgba(234,179,8,0.15);color:#fde047;border:1px solid rgba(234,179,8,0.3);">🟡 Knowledge Base Pending Upload</span>'
)

st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">🤖 AI Document Assistant Pro</div>
    <div class="hero-subtitle">
        Enterprise RAG Workspace combining BM25 Keyword Search & FAISS Vector Embeddings with Gemini AI.
    </div>
    <div>
        {status_pill}
        <span class="badge-pill">🔎 BM25 + FAISS Hybrid</span>
        <span class="badge-pill">✨ Gemini LLM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Helper Functions
# -------------------------

def get_confidence_badge(score):
    high_t = st.session_state.get("conf_high_threshold", 80)
    med_t = st.session_state.get("conf_medium_threshold", 60)
    if score >= high_t:
        return '<span class="badge-high">🟢 High Confidence</span>'
    elif score >= med_t:
        return '<span class="badge-medium">🟡 Medium Confidence</span>'
    else:
        return '<span class="badge-low">🔴 Low Confidence</span>'

# -------------------------
# Sidebar Dashboard & Controls
# -------------------------

st.sidebar.title("🤖 AI Workspace")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.subheader("📁 Knowledge Base Status")
if kb_ready:
    st.sidebar.success("✅ Knowledge Base Active")
else:
    st.sidebar.info("ℹ️ Upload PDFs below to get started")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("📊 Retrieval Analytics")

col_stat1, col_stat2 = st.sidebar.columns(2)
with col_stat1:
    st.metric("Documents", len(st.session_state.get("pdfs", [])))
with col_stat2:
    st.metric("Total Chunks", len(st.session_state.get("chunks", [])))

if "last_retrieved_count" in st.session_state:
    st.sidebar.metric("Last Query Hits", st.session_state["last_retrieved_count"])

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("🎚️ Confidence Thresholds")
st.session_state["conf_high_threshold"] = st.sidebar.slider(
    "High confidence ≥", 0, 100, st.session_state.get("conf_high_threshold", 80)
)
st.session_state["conf_medium_threshold"] = st.sidebar.slider(
    "Medium confidence ≥", 0, st.session_state["conf_high_threshold"],
    min(st.session_state.get("conf_medium_threshold", 60), st.session_state["conf_high_threshold"])
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.subheader("⚙️ Chat Controls")

if st.sidebar.button("🗑 Clear Conversation"):
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []
    st.rerun()

if st.sidebar.button("🗑 Reset Knowledge Base"):
    for key in ("index", "chunks", "pdfs", "last_retrieved_count"):
        st.session_state.pop(key, None)
    st.rerun()

if "messages" in st.session_state and st.session_state["messages"]:
    chat_text = ""
    for msg in st.session_state["messages"]:
        chat_text += f"{msg['role'].upper()}\n{msg['content']}\n\n"

    st.sidebar.download_button(
        "📥 Export Chat Transcript",
        chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

doc_options = ["All Documents"] + st.session_state.get("pdfs", [])
source_filter = st.sidebar.selectbox("🎯 Metadata Filter (Document)", doc_options)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.write("🧠 **Embedding**: `all-MiniLM-L6-v2`")

with st.sidebar.expander("🔧 API Diagnostics"):
    diag = get_key_diagnostic()
    if diag["found"]:
        st.success(f"Key detected via {diag['source']}: `{diag['masked']}`")
    else:
        st.error(f"No API key detected (checked env vars and st.secrets).")

    errors = get_last_model_errors()
    if errors:
        st.caption("Last generation attempt's per-model errors:")
        for model_name, err in errors:
            st.text(f"{model_name}: {err[:200]}")
    else:
        st.caption("No model errors recorded yet — ask a question to populate this.")
st.sidebar.write("🔎 **Keyword Engine**: `BM25Okapi`")
st.sidebar.write("📦 **Vector Store**: `FAISS IndexFlatIP`")
st.sidebar.write("⚡ **Re-ranker**: `ms-marco-MiniLM-L-6-v2`")

if "pdfs" in st.session_state and st.session_state["pdfs"]:
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.subheader("📄 Uploaded Files")
    for pdf in st.session_state["pdfs"]:
        st.sidebar.write(f"✅ {pdf}")

# -------------------------
# Document Ingestion Manager (Collapsible Expander)
# -------------------------

with st.expander("📄 Document Knowledge Base Manager", expanded=not kb_ready):
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents to index into the Knowledge Base",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data/pdfs", exist_ok=True)
        saved_files = []
        skipped_duplicates = []
        already_indexed = set(st.session_state.get("pdfs", []))

        for uploaded_file in uploaded_files:
            safe_name = sanitize_filename(uploaded_file.name)
            if safe_name in already_indexed:
                skipped_duplicates.append(safe_name)
                continue
            save_path = os.path.join("data/pdfs", safe_name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(save_path)

        if skipped_duplicates:
            st.warning(f"⏭ Skipped {len(skipped_duplicates)} already-indexed file(s): {', '.join(skipped_duplicates)}")

        if saved_files:
            st.info(f"📁 {len(saved_files)} PDF document(s) staged for indexing.")

        if saved_files and st.button("📚 Process & Build Knowledge Base"):
            build_start = time.time()
            progress = st.progress(0)

            with st.spinner("Processing & indexing documents..."):
                all_chunks = []
                progress.progress(20)

                for pdf_path in saved_files:
                    pages = load_pdf(pdf_path)
                    chunks = chunk_text(pages, pdf_path)
                    all_chunks.extend(chunks)

                progress.progress(50)
                embeddings = create_embeddings(all_chunks)

                progress.progress(80)
                index = create_index(embeddings)
                save_index(index, all_chunks)
                build_bm25(all_chunks)

                st.session_state["index"] = index
                st.session_state["chunks"] = all_chunks

                existing_pdfs = st.session_state.get("pdfs", [])
                new_pdfs = [os.path.basename(pdf) for pdf in saved_files]
                st.session_state["pdfs"] = existing_pdfs + [
                    p for p in new_pdfs if p not in existing_pdfs
                ]
                progress.progress(100)

            build_end = time.time()
            st.success(f"🎉 Knowledge Base indexed in {build_end - build_start:.2f} seconds!")
            st.rerun()

# -------------------------
# Organized AI Chat Stream
# -------------------------

# Welcome / Getting Started Card when conversation is empty
if not st.session_state["messages"]:
    st.markdown("""
    <div class="welcome-card">
        <h3 style="color: #c084fc; margin-bottom: 0.5rem;">💬 Welcome to your AI Document Assistant</h3>
        <p style="color: #94a3b8; font-size: 0.98rem; max-width: 600px; margin: 0 auto 1.2rem auto;">
            Ask any question about your indexed documents. Answers are grounded in real document context with full citations and page numbers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sample Quick Suggestion Buttons
    col_sug1, col_sug2, col_sug3 = st.columns(3)
    preset_query = None
    with col_sug1:
        if st.button("💡 Summarize key topics"):
            preset_query = "Summarize the key topics and main points in the document."
    with col_sug2:
        if st.button("💡 What are the main requirements?"):
            preset_query = "What are the main requirements and findings mentioned?"
    with col_sug3:
        if st.button("💡 Executive Overview"):
            preset_query = "Provide an executive overview of the uploaded PDF."

    if preset_query:
        st.session_state["messages"].append({"role": "user", "content": preset_query})
        st.rerun()

# Display Previous Conversation Stream
for message in st.session_state["messages"]:
    avatar_icon = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

        # Display Sources as an Organized Accordion Drawer
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 View {len(message['sources'])} Cited Document Sources"):
                for idx, result in enumerate(message["sources"], start=1):
                    badge_html = get_confidence_badge(result['score'])
                    st.markdown(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 12px; margin-bottom: 10px;">
                            <div style="margin-bottom: 6px;">
                                <strong>📄 Citation #{idx}:</strong> <code>{result['source']}</code> &bull; Page {result['page']} &bull; Match: <strong>{result['score']}%</strong> &bull; {badge_html}
                            </div>
                            <div style="font-size: 0.92rem; color: #cbd5e1; font-style: italic; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px;">
                                "{result['text']}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Chat Input Field
prompt_query = st.chat_input("Ask a question about your documents...")

if prompt_query:
    query = prompt_query.strip()
    if not query:
        st.warning("Please enter a question.")
    elif not kb_ready:
        st.error("Please upload a PDF document and click 'Process & Build Knowledge Base' first.")
    else:
        # Append User Message
        st.session_state["messages"].append({"role": "user", "content": query})

        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        start_time = time.time()

        with st.spinner("Searching Knowledge Base (BM25 + FAISS + Cross-Encoder Re-ranking)..."):
            results = hybrid_search(
                query=query,
                model=model,
                index=st.session_state["index"],
                chunks=st.session_state["chunks"],
                top_k=5,
                source_filter=source_filter
            )

        st.session_state["last_retrieved_count"] = len(results)

        with st.spinner("Generating AI Answer with Gemini..."):
            answer, degraded = get_answer(query, results)

        end_time = time.time()

        top_score = max([r['score'] for r in results]) if results else 0
        looks_like_refusal = answer.strip().rstrip(".") == "I don't have enough information"

        # Render Assistant Response
        with st.chat_message("assistant", avatar="🤖"):
            if degraded:
                st.warning(
                    "⚠️ AI summarization is temporarily unavailable (rate limit or generation error). "
                    "Showing the top matching excerpts instead — this is **not** an AI-generated answer.",
                    icon="⚠️"
                )
            elif looks_like_refusal and top_score >= st.session_state.get("conf_high_threshold", 80):
                st.info(
                    "💡 The model couldn't form a direct answer, but the retrieved excerpts below look "
                    "highly relevant — try rephrasing your question more specifically, or check the "
                    "cited sources directly.",
                    icon="💡"
                )

            placeholder = st.empty()
            full_answer = ""
            words = answer.split()

            if degraded:
                # Excerpts are already retrieved content — no need to fake-stream them
                placeholder.markdown(answer)
                full_answer = answer
            else:
                batch_size = 3
                for i in range(0, len(words), batch_size):
                    full_answer += " ".join(words[i:i + batch_size]) + " "
                    placeholder.markdown(full_answer)
                    time.sleep(0.04)

            # Metadata Bar under AI Answer
            best_badge = get_confidence_badge(top_score)
            
            col_act1, col_act2 = st.columns([1, 3])
            with col_act1:
                st.download_button(
                    "📥 Download Answer",
                    answer,
                    file_name="answer.txt",
                    mime="text/plain",
                    key=f"dl_{len(st.session_state['messages'])}"
                )
            with col_act2:
                st.markdown(
                    f"<div style='font-size: 0.85rem; color: #94a3b8; padding-top: 8px;'>"
                    f"⚡ Latency: <strong>{end_time - start_time:.2f}s</strong> &bull; Top Match: {best_badge}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Organized Sources Drawer
            if results:
                with st.expander(f"📚 View {len(results)} Cited Document Sources"):
                    for idx, result in enumerate(results, start=1):
                        badge_html = get_confidence_badge(result['score'])
                        st.markdown(
                            f"""
                            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 12px; margin-bottom: 10px;">
                                <div style="margin-bottom: 6px;">
                                    <strong>📄 Citation #{idx}:</strong> <code>{result['source']}</code> &bull; Page {result['page']} &bull; Match: <strong>{result['score']}%</strong> &bull; {badge_html}
                                </div>
                                <div style="font-size: 0.92rem; color: #cbd5e1; font-style: italic; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px;">
                                    "{result['text']}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        # Persist Messages & History
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "sources": results
            }
        )

        st.session_state["chat_history"].append(
            {
                "question": query,
                "answer": answer
            }
        )

# -------------------------
# Footer
# -------------------------

st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.88rem; margin-top: 3rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 1.5rem;'>"
    "Semantic Search AI System &bull; Built with Streamlit, BM25, FAISS, SentenceTransformers, and Gemini AI"
    "</div>",
    unsafe_allow_html=True
)