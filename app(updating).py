# app.py
import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_objectbox.vectorstores import ObjectBox

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --------------------
# Setup & Config
# --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_DIR = "./us_census"  # folder containing PDFs

st.set_page_config(page_title="ObjectBox + Ollama + Groq Demo", page_icon="🗂️")
st.title("ObjectBox VectorstoreDB with Llama3 (Groq) + Ollama Embeddings")

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
#model_name="us_census_vectors" 
# --------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # Groq's model id (lowercase)
)

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
    # # Embeddings from local Ollama
    # st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 768-d

    # # Load PDFs
    # loader = PyPDFDirectoryLoader(DATA_DIR)
    # st.session_state.docs = loader.load()

    # # Split into chunks
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # st.session_state.final_documents = splitter.split_documents(st.session_state.docs[:50])

    # # Create / refresh ObjectBox vector store
    # st.session_state.vectors = ObjectBox.from_documents(
    #     st.session_state.final_documents,
    #     st.session_state.embeddings,
    #     embedding_dimensions=768,   # match nomic-embed-text
    # )
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    loader = PyPDFDirectoryLoader(DATA_DIR)
    st.session_state.docs = loader.load()
    st.write(f"Loaded {len(st.session_state.docs)} documents")  # debug

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = splitter.split_documents(st.session_state.docs[:50])
    st.write(f"Split into {len(st.session_state.final_documents)} chunks")  # debug

    st.session_state.vectors = ObjectBox.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings,
        embedding_dimensions=768,
        model_name="us_census_vectors" 
    )

col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("Build / Refresh Index"):
        with st.spinner("Building ObjectBox index..."):
            try:
                build_index()
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
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})

    # 2. Run retrieval
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input_prompt})

    # 3. Extra guard: no context retrieved
    if not response.get("context"):
        st.warning("No context retrieved. Try another query or check your PDFs.")
        st.stop()

    # 4. Show results
    st.subheader("Answer")
    st.write(response["answer"])

    with st.expander("Document Similarity Search (Top Chunks)"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---")
