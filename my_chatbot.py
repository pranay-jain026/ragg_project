import os
import time

import requests  # type: ignore
import streamlit as st  # type: ignore
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore
try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    ChatOllama = None  # type: ignore
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception:
    FAISS = None  # type: ignore
try:
    from langchain_ollama import OllamaEmbeddings  # type: ignore
except Exception:
    OllamaEmbeddings = None  # type: ignore
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    RecursiveCharacterTextSplitter = None  # type: ignore
from typing import Any, Sequence


st.set_page_config(page_title="My Chatbot", page_icon="🤖")
st.header("My Chatbot")
st.write("Upload a PDF, then ask questions from it.")


def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF file safely."""
    pdf_reader = PdfReader(uploaded_file)
    pages_text = []
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text).strip()


@st.cache_resource(show_spinner=False)
def build_vector_store(text: str, ollama_base_url: str, embed_model: str) -> Any:
    """Create embeddings and FAISS index (cached for speed) using Ollama."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " "],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_base_url)
    return FAISS.from_texts(chunks, embeddings)


def answer_from_documents(llm: Any, question: str, docs: Sequence[Any]) -> str:
    """Generate an answer grounded in retrieved document chunks."""
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""
You are a helpful assistant answering questions from a PDF.
Use only the provided context. If the answer is not in the context, say that clearly.

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(prompt.strip())
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


def with_retries(func, retries: int = 3, delay_seconds: float = 1.5):
    """Retry transient network calls a few times before failing."""
    last_error = None
    for attempt in range(retries):
        try:
            return func()
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds * (attempt + 1))
            else:
                raise
    raise last_error


def is_quota_error(exc: Exception) -> bool:
    """Detect rate/quota style errors."""
    msg = str(exc).lower()
    return "insufficient_quota" in msg or "you exceeded your current quota" in msg or "429" in msg


def check_ollama_health(ollama_base_url: str, chat_model: str, embed_model: str) -> tuple[bool, str]:
    """Check Ollama server reachability and model availability."""
    try:
        response = requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=8)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        available_names = {
            model.get("name", "").split(":")[0]
            for model in models
            if isinstance(model, dict) and model.get("name")
        }
        missing_models = [m for m in [chat_model, embed_model] if m.split(":")[0] not in available_names]
        if missing_models:
            missing_text = ", ".join(f"`{name}`" for name in missing_models)
            return False, (
                f"Ollama is reachable, but model(s) not found: {missing_text}. "
                f"Run `ollama pull {missing_models[0]}` (and pull others as needed)."
            )
        return True, "Ollama is reachable and selected models are available."
    except requests.exceptions.RequestException as exc:
        return False, f"Could not reach Ollama at `{ollama_base_url}`. Error: {exc}"
    except ValueError:
        return False, "Ollama returned a non-JSON response. Check the base URL."


with st.sidebar:
    st.title("Info")
    st.caption("Use the main page to upload and ask questions with Ollama.")
    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ).strip()
    chat_model = st.text_input(
        "Chat Model",
        value=os.getenv("OLLAMA_CHAT_MODEL", "phi3"),
    ).strip()
    embed_model = st.text_input(
        "Embedding Model",
        value=os.getenv("OLLAMA_EMBED_MODEL", "phi3"),
    ).strip()
    if st.button("Check Ollama Health"):
        if not ollama_base_url:
            st.warning("Set a valid Ollama base URL first.")
        elif not chat_model or not embed_model:
            st.warning("Set both chat and embedding model names first.")
        else:
            ok, message = check_ollama_health(ollama_base_url, chat_model, embed_model)
            if ok:
                st.success(message)
            else:
                st.error(message)

file = st.file_uploader("Upload a PDF file", type="pdf")

deps_missing = any(
    dependency is None
    for dependency in [ChatOllama, OllamaEmbeddings, FAISS, RecursiveCharacterTextSplitter, PdfReader]
)
if deps_missing:
    st.error(
        "Missing required packages. Install: "
        "`pip install streamlit langchain langchain-ollama langchain-community "
        "langchain-text-splitters faiss-cpu pypdf2`"
    )
    st.stop()

if not ollama_base_url:
    st.warning("Set a valid Ollama base URL in the sidebar.")
elif not chat_model or not embed_model:
    st.warning("Set both chat and embedding models in the sidebar.")
else:
    st.caption(f"Using Ollama at `{ollama_base_url}` with chat `{chat_model}` and embeddings `{embed_model}`.")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_file_name" not in st.session_state:
    st.session_state.indexed_file_name = ""

if file is not None:
    st.success(f"Uploaded: {file.name}")
    process_clicked = st.button("Process PDF", type="primary")
    if process_clicked:
        if not ollama_base_url or not embed_model:
            st.warning("Set valid Ollama configuration first, then click `Process PDF`.")
            st.stop()
        try:
            text = extract_pdf_text(file)
        except Exception as exc:
            st.error(f"Could not read the PDF. Error: {exc}")
            st.stop()

        if not text:
            st.warning("No readable text found in this PDF.")
            st.stop()

        try:
            with st.spinner("Indexing your document..."):
                st.session_state.vector_store = with_retries(
                    lambda: build_vector_store(text, ollama_base_url, embed_model)
                )
                st.session_state.indexed_file_name = file.name
            st.success("PDF processed. You can ask questions now.")
        except requests.exceptions.ConnectionError:
            st.error(
                "Network error while contacting Ollama. "
                "Please ensure Ollama is running locally and base URL is correct."
            )
            st.stop()
        except Exception as exc:
            if is_quota_error(exc):
                st.error("Rate limit hit (429). Try again after a short wait.")
            else:
                st.error(f"Could not build document index. Error: {exc}")
            st.stop()
else:
    st.info("Upload a PDF above, then click `Process PDF`.")

st.subheader("Ask Questions")
if st.session_state.indexed_file_name:
    st.caption(f"Current document: {st.session_state.indexed_file_name}")

with st.form("ask_form", clear_on_submit=False):
    user_question = st.text_input(
        "Search / Ask a question from your PDF",
        placeholder="Type your question and press Enter...",
    )
    ask_clicked = st.form_submit_button("Get Answer")

if ask_clicked:
    if not user_question.strip():
        st.warning("Please type a question first.")
    elif not ollama_base_url or not chat_model:
        st.warning("Set valid Ollama configuration first.")
    elif st.session_state.vector_store is None:
        st.warning("Please upload and process a PDF first.")
    else:
        try:
            with st.spinner("Finding answer..."):
                match = with_retries(
                    lambda: st.session_state.vector_store.similarity_search(
                        user_question.strip(), k=4
                    )
                )
                llm = ChatOllama(
                    model=chat_model,
                    base_url=ollama_base_url,
                    temperature=0,
                )
                response = with_retries(
                    lambda: answer_from_documents(llm, user_question.strip(), match)
                )
        except requests.exceptions.ConnectionError:
            st.error(
                "Network error while contacting Ollama. "
                "Please ensure Ollama is running locally and base URL is correct."
            )
            st.stop()
        except Exception as exc:
            if is_quota_error(exc):
                st.error("Rate limit hit (429). Try again after a short wait.")
            else:
                st.error(f"Could not generate answer. Error: {exc}")
            st.stop()

        st.subheader("Answer")
        st.write(response)