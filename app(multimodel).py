# app.py  (WORKING WELL-GOOD)
import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_objectbox.vectorstores import ObjectBox

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pathlib import Path

# --------------------
# Setup & Config
# --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# point to the folder next to this app.py
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "us_sensus"   # folder containing PDFs

st.set_page_config(page_title="ObjectBox + Ollama + Groq Demo", page_icon="🗂️")
st.title("ObjectBox VectorstoreDB with Llama3 (Groq) + Ollama Embeddings")

# --------------------
# Model Selection
# --------------------
llm_choice = st.selectbox(
    "Choose LLM model:",
    [
        "groq:llama-3.1-8b-instant",
        "groq:llama-3.3-70b-versatile",
        "ollama:llama3:8b",
        "ollama:llama2:latest"
    ],
    index=0
)

embed_choice = st.selectbox(
    "Choose Embedding model (rebuild index if changed):",
    [
        "ollama:nomic-embed-text"
        # you can add others later
    ],
    index=0
)

# Initialize session state keys to avoid AttributeError
for key, default in [
    ("vectors", None),
    ("embeddings", None),
    ("final_documents", None),
    ("docs", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------
# LLM (Groq current model)   Here removed because of multimodel
# --------------------
# llm = ChatGroq(
#     groq_api_key=GROQ_API_KEY,
#     model_name="llama-3.1-8b-instant",
# )
# --------------------
# Build LLM based on selection
# --------------------

def build_llm():
    if llm_choice.startswith("groq:"):
        from langchain_groq import ChatGroq
        model_id = llm_choice.split("groq:", 1)[1]
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_id)
    elif llm_choice.startswith("ollama:"):
        from langchain_community.llms import Ollama
        model_id = llm_choice.split("ollama:", 1)[1]
        return Ollama(model=model_id)
    else:
        st.error("Unsupported LLM choice")
        st.stop()

llm = build_llm()



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
def build_index():
    # 1) Embeddings from local Ollama
    # st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 768-d

    if embed_choice.startswith("ollama:"):
        emb_model = embed_choice.split("ollama:", 1)[1]
        st.session_state.embeddings = OllamaEmbeddings(model=emb_model)
    else:
        st.error("Unsupported embedding choice")
        st.stop()

    # 2) Load PDFs
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    st.session_state.docs = loader.load()
    st.write(f"📄 Loaded {len(st.session_state.docs)} PDF documents")

    if not st.session_state.docs:
        st.warning("No PDFs found. Check the path ./us_census")
        return

    # 3) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = splitter.split_documents(st.session_state.docs[:50])
    st.write(f"🧱 Created {len(st.session_state.final_documents)} text chunks")

    if not st.session_state.final_documents:
        st.warning("No text extracted from PDFs. If your PDFs are scanned images, use an OCR loader.")
        return

    # 4) Create / refresh ObjectBox vector store
    st.session_state.vectors = ObjectBox.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings,
        embedding_dimensions=768,   # match nomic-embed-text
       # model_name="us_census_vectors"   # avoid 'VectorEntity' replacement warning
    )

col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("Build / Refresh Index"):
        with st.spinner("Building ObjectBox index..."):
            try:
                build_index()
                if st.session_state.vectors:
                    st.success("ObjectBox Database is ready ✅")
            except Exception as e:
                st.session_state.vectors = None
                st.error(f"Index build failed: {e}")

with col_b:
    st.info("Steps: 1) Click **Build / Refresh Index**  2) Ask a question below")

# --------------------
# Query UI
# --------------------
input_prompt = st.text_input("Enter your question about the documents:")

if input_prompt:
    if not st.session_state.vectors:
        st.warning("Please build the index first (click **Build / Refresh Index**).")
        st.stop()

    # 1. Build the chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Use a mild threshold & ask for a few more candidates
    retriever = st.session_state.vectors.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.2, "k": 8}
    )

    # 2. Run retrieval
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": input_prompt})
    elapsed = time.process_time() - start

    # 3. Extra guard: no context retrieved
    if not response.get("context"):
        st.warning("No context retrieved. Try another query or check your PDFs.")
        # ---- Debug: show what vector search returns directly
        st.info("Debugging: raw similarity scores")
        hits = st.session_state.vectors.similarity_search_with_score(input_prompt, k=5)
        if not hits:
            st.write("No hits from vector search either.")
        else:
            for d, score in hits:
                st.write(f"Score: {score:.4f}")
                st.write(d.page_content[:400])
                st.write("---")
        st.stop()

    # 4. Show results
    st.write(f"**Response time:** {elapsed:.2f}s")
    st.subheader("Answer")
    st.write(response["answer"])

    with st.expander("Document Similarity Search (Top Chunks)"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---")
