import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables (local dev — .env should never be committed to the repo)
load_dotenv()

def _get_api_key():
    """Check plain env vars first (local .env / OS env), then fall back to
    Streamlit's secrets manager (used on Streamlit Community Cloud, where
    there is no .env file — keys are set via the app's Settings > Secrets).
    Returns (key_or_None, source_label) for diagnostics."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key, "environment variable"
    try:
        key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        if key:
            return key, "st.secrets"
    except Exception:
        pass
    return None, "not found"

# Get API key from .env / OS env (supports GOOGLE_API_KEY and GEMINI_API_KEY),
# falling back to Streamlit Cloud secrets if not found.
api_key, api_key_source = _get_api_key()

if api_key:
    genai.configure(api_key=api_key)

def get_key_diagnostic():
    """Masked, safe-to-display diagnostic for the sidebar — never exposes the
    actual key."""
    if not api_key:
        return {"found": False, "source": api_key_source, "masked": None}
    masked = f"{api_key[:4]}…{api_key[-4:]}" if len(api_key) > 8 else "****"
    return {"found": True, "source": api_key_source, "masked": masked}

def get_last_model_errors():
    """Returns the list of (model_name, error_message) pairs from the most
    recent generate_answer() call that hit the fallback, for debugging."""
    return _last_model_errors

# Candidate Gemini model list for failover.
# NOTE (updated 2026-07-25): gemini-1.5-flash / gemini-1.5-pro were fully
# shut down 2025-09-29, and gemini-2.0-flash was shut down 2026-06-01 — all
# three now 404 immediately. Keeping retired models in this list wastes a
# full tenacity retry cycle (3 attempts x up to 6s backoff) on each of them
# before ever reaching a live model, which is what produced the ~34s
# "temporarily unavailable" fallback. Live models only, most current first.
# gemini-2.5-flash is kept as a last resort — it's intermittently returning
# early 404s ahead of its official Oct 16, 2026 shutdown date.
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]

# Marker prefix used to detect degraded (non-LLM) responses from app.py
# without relying on string-matching exception messages.
FALLBACK_MARKER = "AI Summarization temporarily unavailable"


def generate_extractive_fallback(question, chunks):
    """
    Styled, graceful UI notice when AI API Quota (429) is temporarily exhausted,
    or when every candidate model is unavailable.
    Exhibits clean presentation of top matched excerpts.
    """
    if not chunks:
        return "I don't have enough information."

    res_text = f"{FALLBACK_MARKER}. Showing top matched excerpts below:\n\n"
    for i, c in enumerate(chunks, start=1):
        source = c.get("source", "Document")
        page = c.get("page", 1)
        text = c.get("text", c.get("chunk", ""))
        # Bold instead of backtick/code formatting for the filename — backticks
        # render as <code>, which needs its own theme styling to avoid showing
        # up as a stray white box on dark UIs.
        res_text += f"**Excerpt {i}** (**{source}**, Page {page}):\n> {text}\n\n"

    return res_text.strip()


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True
)
def call_gemini_with_retry(model_name, prompt):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    if response and hasattr(response, "text") and response.text:
        return response.text
    raise ValueError("Empty response text from model")


_last_model_errors = []

def generate_answer(question, chunks):
    """
    Production RAG Answer Generation:
    1. Grounded System Prompt with structural boundaries
    2. Multi-Model Candidate Failover
    3. Tenacity Exponential Backoff Retry Handling
    4. Graceful Extractive Fallback for Rate Limits
    """
    global _last_model_errors
    _last_model_errors = []

    if not chunks:
        return "I don't have enough information."

    formatted_chunks = []
    for i, c in enumerate(chunks, start=1):
        source = c.get("source", "Document")
        page = c.get("page", 1)
        text = c.get("text", c.get("chunk", ""))
        formatted_chunks.append(f"[Source {i}: {source}, Page {page}]\n{text}")

    context_str = "\n\n---\n\n".join(formatted_chunks)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the CONTEXT below —
do not use outside knowledge. The question may be phrased broadly or differently from the wording
in the context (e.g. "what are the topics", "summarize this", "give an overview") — in that case,
synthesize, summarize, or infer an answer by reading across the context rather than looking for an
exact phrase match. Only respond with 'I don't have enough information.' if the context truly
contains nothing relevant to what's being asked.

CONTEXT:
{context_str}

USER QUESTION:
{question}

ANSWER:
"""

    if not api_key:
        _last_model_errors.append(("(no model attempted)", f"No API key found (source: {api_key_source})"))
        return generate_extractive_fallback(question, chunks)

    # Multi-model fallback architecture with tenacity retries
    for model_name in GEMINI_MODELS:
        try:
            return call_gemini_with_retry(model_name, prompt)
        except Exception as e:
            # Failover list: try the next candidate regardless of failure
            # reason (quota, 404/retired model, transient network error, etc.)
            _last_model_errors.append((model_name, str(e)))
            continue

    # Graceful fallback if every candidate model failed
    return generate_extractive_fallback(question, chunks)