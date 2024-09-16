import streamlit as st
from src.rag_core import VectorStoreManager, RAGChain
from src.utils import load_docs_from_jsonl

# Example Streamlit app code to interact with your RAG system
st.title("Personal Travel Assistant")

# Initialize RAG components
vector_store_manager = VectorStoreManager()

# Load documents and setup vector store
chunked_documents = load_docs_from_jsonl(
    "data/processed_dataset/chunked_documents.jsonl"
)
raw_documents = load_docs_from_jsonl("data/processed_dataset/raw_documents.jsonl")
vector_store_manager.ingest_documents(chunked_documents, raw_documents)
retriever = vector_store_manager.get_retriever()

# Initialize RAG chain with user's choice of LLM
llm_choice = st.selectbox("Choose an LLM:", ["groq", "openai"])
rag_chain = RAGChain(retriever=retriever, llm_type=llm_choice)

# Question input and response generation
question = st.text_input("Ask a question about your travel document:")
if question:
    answer = rag_chain.answer_question(question)
    st.write(f"Answer: {answer}")
