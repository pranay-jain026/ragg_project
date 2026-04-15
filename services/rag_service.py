from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vector_store(text, base_url, embed_model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_text(text)

    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url=base_url
    )

    return FAISS.from_texts(chunks, embeddings)


def answer_from_documents(llm, question, docs):
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Use only this context:

{context}

Question: {question}
"""

    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)