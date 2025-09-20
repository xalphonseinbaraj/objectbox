# app.py
import os
import time
import shutil
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_objectbox.vectorstores import ObjectBox

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --------------------
# Setup
# --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "us_sensus"   # <— adjust to your folder (e.g., us_sensus)

loader = PyPDFDirectoryLoader(str(DATA_DIR), glob="**/*.pdf")
docs = loader.load()
st.info(f"Loaded {len(docs)} PDF documents from: {DATA_DIR}")


st.set_page_config(page_title="RAG: ObjectBox + Ollama + Groq", page_icon="🗂️")
st.title("RAG with ObjectBox + Ollama Embeddings + LLM Picker")

# --------------------
# Session init (avoid KeyErrors)
# --------------------
for key, default in [
    ("vectors", None),
    ("embeddings", None),
    ("final_documents", None),
    ("docs", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------
# Model pickers (before any button)
# --------------------
st.subheader("1) Choose Models")

llm_choice = st.selectbox(
    "LLM model:",
    [
        "groq:llama-3.1-8b-instant",
        "groq:llama-3.3-70b-versatile",
        "ollama:llama3:8b",
        "ollama:llama2:latest",
    ],
    index=0,
    key="llm_choice",
)

embed_choice = st.selectbox(
    "Embedding model (requires rebuild):",
    [
        "ollama:nomic-embed-text",
        "openai:text-embedding-3-small",
        "openai:text-embedding-3-large",
    ],
    index=0,
    key="embed_choice",
)

# Show current selections so you can see Streamlit remembers them
st.caption(f"Selected LLM: **{llm_choice}**  |  Embedding: **{embed_choice}**")

# --------------------
# Build LLM based on selection
# --------------------
def build_llm():
    if llm_choice.startswith("groq:"):
        from langchain_groq import ChatGroq
        model_id = llm_choice.split("groq:", 1)[1]
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_id)

    if llm_choice.startswith("ollama:"):
        from langchain_community.llms import Ollama
        model_id = llm_choice.split("ollama:", 1)[1]
        return Ollama(model=model_id)

    st.error("Unsupported LLM choice")
    st.stop()

llm = build_llm()

# --------------------
# Prompt Template
# --------------------
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Be precise and concise.

<context>
{context}
</context>

Question: {input}
"""
)

# --------------------
# Build / Refresh Index
# --------------------
st.subheader("2) Build / Refresh Index")

def build_index():
    # Optional: reset ObjectBox storage for a clean build
    db_path = Path("objectbox")
    if db_path.exists():
        shutil.rmtree(db_path)
        st.info("🧹 Cleared old ObjectBox database")

    # Embeddings
    if embed_choice.startswith("ollama:"):
        emb_model = embed_choice.split("ollama:", 1)[1]
        st.session_state.embeddings = OllamaEmbeddings(model=emb_model)
        emb_dim = 768  # nomic-embed-text vectors are 768-d

    elif embed_choice.startswith("openai:"):
        from langchain_openai import OpenAIEmbeddings
        emb_model = embed_choice.split("openai:", 1)[1]
        st.session_state.embeddings = OpenAIEmbeddings(model=emb_model)
        emb_dim = 1536 if "small" in emb_model else 3072

    else:
        st.error("Unsupported embedding choice")
        st.stop()

    # Load PDFs
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    st.session_state.docs = loader.load()
    st.info(f"Loaded {len(st.session_state.docs)} PDF documents from: {DATA_DIR}")

    if not st.session_state.docs:
        st.warning(f"No PDFs found. Check the folder path.")
        return

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = splitter.split_documents(st.session_state.docs[:50])
    st.success(f"Created {len(st.session_state.final_documents)} text chunks")

    if not st.session_state.final_documents:
        st.warning("No text extracted from PDFs. Are they scanned images? (Use OCR loader.)")
        return

    # Create vector store
    st.session_state.vectors = ObjectBox.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings,
        embedding_dimensions=emb_dim,
    )

# Button to build the index AFTER both selections are visible
if st.button("Build / Refresh Index", use_container_width=True):
    with st.spinner("Building ObjectBox index..."):
        try:
            build_index()
            if st.session_state.vectors:
                st.success("✅ ObjectBox index is ready")
        except Exception as e:
            st.session_state.vectors = None
            st.error(f"Index build failed: {e}")

st.markdown("---")

# --------------------
# Query UI
# --------------------
st.subheader("3) Ask a question")

input_prompt = st.text_input("Enter your question about the documents:")

if input_prompt:
    if not st.session_state.vectors:
        st.warning("Please build the index first (click **Build / Refresh Index**).")
        st.stop()

    # Build the chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # A bit more forgiving retriever for early tests
    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 3}
        # or:
        # search_type="similarity_score_threshold",
        # search_kwargs={"score_threshold": 0.2, "k": 8}
    )

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Run retrieval
    start = time.process_time()
    response = retrieval_chain.invoke({"input": input_prompt})
    elapsed = time.process_time() - start

    # Guard: no context retrieved
    if not response.get("context"):
        st.warning("⚠️ No context retrieved. Try another query or check your PDFs.")

        # Debug: show raw vector hits with scores
        st.info("Debugging: raw similarity scores")
        hits = st.session_state.vectors.similarity_search_with_score(input_prompt, k=5)
        if not hits:
            st.write("No hits from vector search either.")
        else:
            for d, score in hits:
                st.write(f"Score: {score:.4f}")
                st.write(d.page_content[:300])
                st.write("---")
        st.stop()

    # Show results
    st.write(f"**Response time:** {elapsed:.2f}s")
    st.subheader("Answer")
    st.write(response["answer"])

    with st.expander("Document Similarity Search (Top Chunks)"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---")
