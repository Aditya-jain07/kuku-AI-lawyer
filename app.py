"""Streamlit interface for Kuku AI Legal & Citizen Assistance."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import re
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import streamlit as st

from config import settings
from src.assistant_service import AssistantService, MAX_QUERY_CHARS
from src.groq_client import MAX_AUDIO_BYTES
from src.models import CaseState
from src.safety import source_label


logger = logging.getLogger(__name__)
CHAT_HISTORY_PATH = Path.home() / ".kuku" / "chats.sqlite3"
AUDIO_EXTENSIONS = ("flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm")
AUDIO_UPLOAD_MB = MAX_AUDIO_BYTES // (1024 * 1024)
st.set_page_config(page_title="Kuku AI Legal Assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ Kuku AI Legal & Citizen Assistance")
st.caption("Legal information for common Indian citizen problems. Kuku is not a substitute for a qualified lawyer.")


def _index_version() -> tuple[Any, ...]:
    """Invalidate the cached service when the local index generation changes."""
    version: list[Any] = [str(settings.vector_store_dir.resolve())]
    for name in ("index_manifest.json", "index.faiss", "metadata.jsonl", "metadata.json", ".index-build.lock"):
        try:
            stat = (settings.vector_store_dir / name).stat()
            version.append((name, stat.st_mtime_ns, stat.st_size, stat.st_ino))
        except FileNotFoundError:
            version.append((name, None))
    return tuple(version)


@st.cache_resource(show_spinner=False, max_entries=2)
def _service_resource(index_version: tuple[Any, ...]) -> tuple[AssistantService, Any]:
    # cache_resource shares instances across sessions. Retriever has per-query
    # mutable flags, so callers must serialize access to this shared service.
    return AssistantService(settings), threading.RLock()


def get_service() -> AssistantService:
    """Preserve the original helper; UI operations use its resource lock too."""
    return _service_resource(_index_version())[0]


_DEFAULTS = {
    "turn_sources": {}, "last_audio_hash": None, "audio_input_version": 0,
    "voice_error": False, "voice_error_message": "", "last_transcript": "",
    "failed_text": None, "last_error": "", "pending_delete_chat": None,
}


class ChatDeletedError(Exception):
    """A different tab deleted the chat while it was open."""


def _history_connection() -> sqlite3.Connection:
    CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CHAT_HISTORY_PATH)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _save_active_chat() -> None:
    case = st.session_state.case
    if not case.messages:
        return
    first_question = next((message["content"] for message in case.messages
                           if message["role"] == "user"), "Conversation")
    title = " ".join(first_question.split())
    if len(title) > 48:
        title = title[:47].rstrip() + "…"

    # Save only the citation fields used to render past turns.
    sources = {
        index: [
            {"citation_id": source.get("citation_id"),
             "source_label": source.get("source_label") or source_label(source),
             "source_url": _safe_source_url(source.get("source_url"))}
            for source in rows
        ]
        for index, rows in st.session_state.turn_sources.items()
    }
    payload = json.dumps({"case": case.to_dict(), "sources": sources},
                         ensure_ascii=False, allow_nan=False)
    chat_id = st.session_state.chat_id
    saved_revision = st.session_state.saved_revision
    with closing(_history_connection()) as connection, connection:
        if saved_revision is None:
            connection.execute(
                "INSERT INTO chats VALUES (?, ?, ?, ?)",
                (chat_id, title, case.updated_at, payload),
            )
        else:
            updated = connection.execute(
                """UPDATE chats SET title = ?, updated_at = ?, payload = ?
                   WHERE id = ? AND updated_at = ?""",
                (title, case.updated_at, payload, chat_id, saved_revision),
            )
            if updated.rowcount == 0:
                if connection.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is None:
                    raise ChatDeletedError
                # Keep both conversations if another tab has saved a newer turn.
                chat_id = uuid4().hex
                connection.execute(
                    "INSERT INTO chats VALUES (?, ?, ?, ?)",
                    (chat_id, title, case.updated_at, payload),
                )
    st.session_state.chat_id = chat_id
    st.session_state.saved_revision = case.updated_at


def _try_save_chat() -> bool:
    try:
        _save_active_chat()
    except ChatDeletedError:
        _activate_chat(uuid4().hex, CaseState(), {}, None)
        st.session_state.history_error = "This chat was deleted in another tab."
        return True
    except (OSError, sqlite3.Error, ValueError, TypeError):
        logger.exception("Could not save chat history")
        st.session_state.history_error = "Couldn’t save this chat. It is still open in this tab."
        return False
    st.session_state.history_error = ""
    return True


def _switch_chat(chat_id: str | None = None) -> None:
    """Prepare the next case before resetting the current session."""
    current = st.session_state.case
    if current.messages and current.updated_at != st.session_state.saved_revision:
        if not _try_save_chat():
            return

    if chat_id is None:
        next_case, sources, revision = CaseState(), {}, None
        chat_id = uuid4().hex
    else:
        with closing(_history_connection()) as connection:
            row = connection.execute(
                "SELECT payload, updated_at FROM chats WHERE id = ?", (chat_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Saved chat was removed")
        payload = json.loads(row[0])
        next_case = CaseState.from_dict(payload["case"])
        sources = {int(index): rows for index, rows in payload["sources"].items()}
        revision = row[1]

    _activate_chat(chat_id, next_case, sources, revision)
    st.rerun()


def _activate_chat(chat_id: str, case: CaseState, sources: dict, revision: str | None) -> None:
    next_version = st.session_state.audio_input_version + 1
    for key, value in _DEFAULTS.items():
        st.session_state[key] = copy.deepcopy(value)
    st.session_state.case = case
    st.session_state.turn_sources = sources
    st.session_state.chat_id = chat_id
    st.session_state.saved_revision = revision
    st.session_state.audio_input_version = next_version
    st.session_state.history_error = ""


def _delete_chat(chat_id: str) -> None:
    with closing(_history_connection()) as connection, connection:
        connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    if chat_id == st.session_state.chat_id:
        _activate_chat(uuid4().hex, CaseState(), {}, None)
    st.session_state.pending_delete_chat = None
    st.rerun()


def _show_history() -> None:
    st.header("Chats")
    if st.button("＋ New chat", key="new_chat", use_container_width=True):
        _switch_chat()
    try:
        with closing(_history_connection()) as connection:
            chats = connection.execute(
                "SELECT id, title FROM chats ORDER BY updated_at DESC, id",
            ).fetchall()
        if not chats:
            st.caption("Completed conversations will appear here.")
        for chat_id, title in chats:
            # Titles are user text. Escape Markdown in the button label.
            label = re.sub(r"([\\`*_\[\]<>~])", r"\\\1", title)
            open_column, delete_column = st.columns([4, 1])
            if open_column.button(label, key=f"chat_{chat_id}", use_container_width=True,
                                  disabled=chat_id == st.session_state.chat_id):
                _switch_chat(chat_id)
            if delete_column.button("🗑️", key=f"delete_chat_{chat_id}", help="Delete this chat"):
                st.session_state.pending_delete_chat = chat_id
            if st.session_state.pending_delete_chat == chat_id:
                st.caption("Permanently delete this saved chat?")
                confirm, cancel = st.columns(2)
                if confirm.button("Delete", key=f"confirm_delete_{chat_id}"):
                    _delete_chat(chat_id)
                if cancel.button("Cancel", key=f"cancel_delete_{chat_id}"):
                    st.session_state.pending_delete_chat = None
                    st.rerun()
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, AttributeError):
        logger.exception("Could not read chat history")
        st.warning("Couldn’t open saved chat history.")
    if st.session_state.history_error:
        st.warning(st.session_state.history_error)
        case = st.session_state.case
        if case.messages and case.updated_at != st.session_state.saved_revision:
            if st.button("Retry saving chat", key="retry_save_chat") and _try_save_chat():
                st.rerun()
    st.caption("Chats are saved on this computer for this local Kuku installation.")




def _show_sidebar() -> None:
    _show_history()
    st.divider()
    if settings.has_groq_key:
        st.success("Groq API key configured")
    else:
        st.warning("Groq API key missing")
        st.caption("Configure GROQ_API_KEY in the private .env file and restart the app.")
    st.divider()
    st.subheader("Supported topics")
    st.markdown("""
- Consumer complaints and refunds
- Rental and property issues
- Employment and unpaid salary
- Banking and UPI disputes
- Cyber fraud
- Free legal aid
- Family matters and child protection
- Education, disability and senior-citizen rights
""")


def _safe_source_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    try:
        parsed = urlsplit(value)
        if (parsed.scheme.lower() in {"https", "http"} and parsed.hostname
                and parsed.username is None and parsed.password is None
                and not any(char.isspace() for char in value)):
            return value
    except ValueError:
        pass
    return None


def display_sources(sources: list[dict]) -> None:
    """Display citations without treating a source URL as proof of authenticity."""
    if not sources:
        return
    with st.expander(f"View supporting sources ({len(sources)})", expanded=False):
        for number, source in enumerate(sources):
            citation_id = str(source.get("citation_id") or "")
            if not re.fullmatch(r"C[1-9][0-9]*", citation_id):
                citation_id = "Source"
            st.markdown(f"**[{citation_id}]**")
            # Plain text prevents document metadata from adding Markdown links.
            st.text(str(source.get("source_label") or source_label(source)))
            url = _safe_source_url(source.get("source_url"))
            if url:
                st.link_button("Open source document", url)
            if number < len(sources) - 1:
                st.divider()


def _error_message(exc: Exception, *, audio: bool = False) -> str:
    """Use fixed messages rather than exposing paths, request bodies or secrets."""
    detail = str(exc).casefold()
    if "rate limit" in detail:
        return "The service usage limit was reached. Please wait and try again."
    if "timed out" in detail:
        return "The request timed out. Please try again."
    if "token limit" in detail:
        return "The answer was cut short and was not shown. Please ask a narrower question."
    if "being rebuilt" in detail or "changed during loading" in detail:
        return "The document collection is being updated. Please try again shortly."
    if "authentication" in detail or "api_key" in detail:
        return "The service configuration needs attention. Please check the private API-key settings."
    if "configured groq model" in detail or "model permissions" in detail:
        return "The selected AI model is unavailable for this account. Check LLM_MODEL and restart the app."
    if any(term in detail for term in (
        "missing index artifact", "index generation is incomplete", "checksum",
        "metadata.jsonl", "unsupported index manifest", "configured embedding",
    )):
        return "The document index needs rebuilding. Stop the app, rebuild and validate the index, then restart."
    if audio and ("audio exceeds" in detail or "request size" in detail):
        return "The recording is too large. Please use a shorter recording."
    return ("Audio could not be transcribed. Try again or record a clearer message."
            if audio else "Kuku could not complete this request. Please try again.")


def process_text(text: str) -> bool:
    """Commit a complete turn only; keep the original case intact on failure."""
    text = text.strip()
    if not text:
        return False
    st.session_state.failed_text = text
    st.session_state.last_error = ""
    try:
        # Work on a copy so cancellation/errors cannot leave partial messages
        # or changes to other CaseState fields in the live session.
        working_case = copy.deepcopy(st.session_state.case)
        original_count = len(working_case.messages)
        with thread:
            with st.chat_message("user"):
                st.markdown(text)
            with st.chat_message("assistant"), st.spinner("Finding relevant information..."):
                service, lock = _service_resource(_index_version())
                with lock:
                    result = service.handle_text(text, working_case)
        if result.status == "index_not_ready":
            st.session_state.last_error = result.assistant_text
            return False
        if len(working_case.messages) == original_count:
            st.session_state.last_error = result.assistant_text
            st.session_state.failed_text = None
            return False
        assistant_index = len(working_case.messages) - 1
        if working_case.messages[assistant_index].get("role") != "assistant":
            raise RuntimeError("Assistant service did not complete the conversation turn")
        sources = dict(st.session_state.turn_sources)
        sources[assistant_index] = result.sources or []
        st.session_state.case = working_case
        st.session_state.turn_sources = sources
        st.session_state.failed_text = None
        st.session_state.last_error = ""
        # Clear the current recorder after a successful turn. The transcript
        # remains visible, and unrelated reruns cannot resubmit the recording.
        st.session_state.audio_input_version += 1
        st.session_state.last_audio_hash = None
        st.session_state.voice_error = False
        st.session_state.voice_error_message = ""
    except Exception as exc:
        logger.exception("Failed to process legal assistant turn")
        st.session_state.last_error = _error_message(exc)
        return False
    _try_save_chat()
    # Streamlit's rerun is control flow; keep it outside the error handler.
    st.rerun()
    return True


if "case" not in st.session_state:
    st.session_state.case = CaseState()
for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)
if "chat_id" not in st.session_state:
    st.session_state.chat_id = uuid4().hex
    st.session_state.saved_revision = None
    st.session_state.history_error = ""
    _try_save_chat()  # Preserve an existing open chat after a live script reload.

with st.sidebar:
    _show_sidebar()

thread = st.container()
with thread:
    for message_index, message in enumerate(st.session_state.case.messages):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.markdown(str(message.get("content") or ""))
            if role == "assistant":
                display_sources(st.session_state.turn_sources.get(message_index, []))

feedback = st.empty()

st.divider()
st.subheader("Voice input", anchor=False)
if st.session_state.last_transcript:
    with st.expander("Last transcription", expanded=False):
        st.text(st.session_state.last_transcript)

voice_method = st.radio(
    "How would you like to add audio?",
    ("Record with microphone", "Upload a recording"),
    horizontal=True,
    key="voice_method",
)
voice_version = st.session_state.audio_input_version
if voice_method == "Record with microphone":
    audio = st.audio_input(
        "Speak your legal problem",
        key=f"voice_input_{voice_version}",
        disabled=not settings.has_groq_key,
    )
else:
    # Older Streamlit releases use the server-wide upload limit.
    upload_options = {}
    if "max_upload_size" in inspect.signature(st.file_uploader).parameters:
        upload_options["max_upload_size"] = AUDIO_UPLOAD_MB
    audio = st.file_uploader(
        f"Choose a recording from your device (max {AUDIO_UPLOAD_MB} MB)",
        type=AUDIO_EXTENSIONS,
        key=f"voice_upload_{voice_version}",
        disabled=not settings.has_groq_key,
        help="Selecting a file sends it for transcription.",
        **upload_options,
    )

if audio is not None and settings.has_groq_key:
    if audio.size == 0:
        st.error("This recording is empty. Choose or record another one.")
    elif audio.size > MAX_AUDIO_BYTES:
        st.error(f"This recording exceeds {AUDIO_UPLOAD_MB} MB. Use a shorter recording.")
    else:
        audio_bytes = audio.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            # Claim the recording before the request so a rerun cannot resubmit it.
            st.session_state.last_audio_hash = audio_hash
            st.session_state.voice_error = True
            st.session_state.voice_error_message = "The transcription did not complete. Please retry."
            transcript = None
            try:
                with st.spinner("Transcribing your audio..."):
                    service, lock = _service_resource(_index_version())
                    with lock:
                        transcript = service.transcribe(audio_bytes, audio.name or "voice_input.wav").strip()
                    if not transcript:
                        raise ValueError("No transcription text was returned")
                st.session_state.last_transcript = transcript
                st.session_state.voice_error = False
                st.session_state.voice_error_message = ""
            except Exception as exc:
                logger.exception("Voice transcription failed")
                st.session_state.voice_error_message = _error_message(exc, audio=True)
            if not st.session_state.voice_error and transcript:
                # Reuse the text path to show the transcript in the chat
                # before retrieval starts and save it in the active chat.
                process_text(transcript)

        if st.session_state.voice_error:
            st.error(st.session_state.voice_error_message)
            if st.button("Retry transcription"):
                st.session_state.last_audio_hash = None
                st.session_state.voice_error = False
                st.session_state.voice_error_message = ""
                st.rerun()

text = st.chat_input(
    "Describe your legal problem in English, Hindi or Hinglish",
    max_chars=MAX_QUERY_CHARS, disabled=not settings.has_groq_key,
)
if text:
    process_text(text)

with feedback.container():
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    if st.session_state.failed_text and settings.has_groq_key:
        st.caption("Your last message was not added to the case.")
        if st.button("Retry last message"):
            process_text(st.session_state.failed_text)
