import os
import streamlit as st
import requests

from data.pdf_loader import extract_pdf_text
from services.rag_service import build_vector_store, answer_from_documents
from services.ollama_service import get_llm, check_ollama_health
from utils.retry import with_retries, is_quota_error


st.set_page_config(page_title="My Chatbot", page_icon="🤖")
st.header("My Chatbot")
st.write("Upload a PDF, then ask questions from it.")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Info")

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
        ok, msg = check_ollama_health(ollama_base_url, chat_model, embed_model)
        if ok:
            st.success(msg)
        else:
            st.error(msg)


# ---------------- SESSION STATE ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_file_name" not in st.session_state:
    st.session_state.indexed_file_name = ""


# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload a PDF file", type="pdf")

if file is not None:
    st.success(f"Uploaded: {file.name}")

    if st.button("Process PDF", type="primary"):
        try:
            text = extract_pdf_text(file)

            if not text:
                st.warning("No readable text found.")
                st.stop()

            with st.spinner("Indexing your document..."):
                st.session_state.vector_store = with_retries(
                    lambda: build_vector_store(text, ollama_base_url, embed_model)
                )
                st.session_state.indexed_file_name = file.name

            st.success("PDF processed. You can ask questions now.")

        except requests.exceptions.ConnectionError:
            st.error("Ollama not running. Check URL.")
        except Exception as exc:
            if is_quota_error(exc):
                st.error("Rate limit hit (429). Try again.")
            else:
                st.error(f"Error: {exc}")


# ---------------- ASK QUESTIONS ----------------
st.subheader("Ask Questions")

if st.session_state.indexed_file_name:
    st.caption(f"Current document: {st.session_state.indexed_file_name}")

query = st.text_input("Search / Ask a question from your PDF")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Enter a question.")
    elif st.session_state.vector_store is None:
        st.warning("Upload and process a PDF first.")
    else:
        try:
            with st.spinner("Finding answer..."):
                docs = with_retries(
                    lambda: st.session_state.vector_store.similarity_search(query, k=4)
                )

                llm = get_llm(chat_model, ollama_base_url)

                response = with_retries(
                    lambda: answer_from_documents(llm, query, docs)
                )

            st.subheader("Answer")
            st.write(response)

        except requests.exceptions.ConnectionError:
            st.error("Ollama connection error.")
        except Exception as exc:
            if is_quota_error(exc):
                st.error("Rate limit hit.")
            else:
                st.error(f"Error: {exc}")