import streamlit as st
import os
from src.utils import load_docs_from_jsonl
from src.rag_core import VectorStoreManager, RAGChain
from streamlit_pdf_viewer import pdf_viewer


# Function to get list of PDFs from the data/ folder
def list_pdf_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(".pdf")]


# Function to read PDF file content
def read_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    return pdf_data


# Set folder path for PDFs
folder_path = "data/"

# App Description
st.title("Personal Travel Assistant")
st.markdown(
    """
This application helps you retrieve relevant details from your travel-related documents (such as hotel bookings, flight details, and activities). 
You can also browse and view your travel documents (PDFs) directly within the app.

**Note:** If you choose the OpenAI model, you must input your OpenAI API key to enable the question input.
"""
)

# Create tabs for PDF viewer and chat functionality
tab1, tab2 = st.tabs(["Ask Questions", "PDF Viewer"])

with tab1:
    st.subheader("Ask a Question About the Trip")

    # Initialize RAG components
    doc_loader = load_docs_from_jsonl
    vector_store_manager = VectorStoreManager()

    # Load documents and set up vector store
    chunked_documents = doc_loader("data/processed_dataset/chunked_documents.jsonl")
    raw_documents = doc_loader("data/processed_dataset/raw_documents.jsonl")
    vector_store_manager.ingest_documents(chunked_documents, raw_documents)
    retriever = vector_store_manager.get_retriever()

    # Set up RAG chain with user's choice of LLM
    llm_choice = st.selectbox("Choose an LLM:", ["groq", "openai"])

    # OpenAI API Key Input (Only required if the OpenAI model is chosen)
    openai_api_key = None
    if llm_choice == "openai":
        openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to use the OpenAI model.")
            st.stop()  # Stop the app until the API key is provided

    rag_chain = RAGChain(retriever=retriever, llm_type=llm_choice)

    # Question input and response generation
    question = st.text_input("Ask your question:")
    if question:
        answer = rag_chain.answer_question(question)
        st.write(f"Answer: {answer}")

with tab2:
    st.subheader("View Travel Documents (PDFs)")

    # Get list of PDF files
    pdf_files = list_pdf_files(folder_path)

    # Show dropdown to select a PDF file
    selected_pdf = st.selectbox(
        "Select a PDF to view:",
        pdf_files,
        index=None,
        placeholder="Select a file to view...",
    )

    if selected_pdf:
        # Full file path
        file_path = os.path.join(folder_path, selected_pdf)

        # Read the selected PDF file
        pdf_data = read_pdf(file_path)

        # Display the PDF as a download button
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name=selected_pdf,
            mime="application/octet-stream",
        )

        # Display the PDF directly in the app (embed it)
        st.markdown(f"Viewing: {selected_pdf}")
        pdf_viewer(pdf_data)
