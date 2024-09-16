import sys
import pysqlite3 as sqlite3

sys.modules["sqlite3"] = sqlite3

import json
from typing import Iterable
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from src.utils import load_docs_from_jsonl

# Load environment variables (e.g., for GROQ API key)
load_dotenv()


# Manager to handle the vector store (MultiVectorStore)
class VectorStoreManager:
    def __init__(self, embeddings_model: str = "text-embedding-3-small"):
        self.embeddings_model = embeddings_model
        self.vectorstore = Chroma(
            collection_name="full_documents",
            embedding_function=OpenAIEmbeddings(model=self.embeddings_model),
        )
        self.store = InMemoryByteStore()
        self.retriever = None

    def ingest_documents(
        self,
        chunked_documents: Iterable[Document],
        raw_documents: Iterable[Document],
        id_key: str = "doc_id",
    ):
        # Create MultiVectorRetriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.store,
            id_key=id_key,
        )

        # Add documents to the vector store
        doc_ids = [doc.metadata["doc_id"] for doc in raw_documents]
        self.retriever.vectorstore.add_documents(chunked_documents)
        self.retriever.docstore.mset(list(zip(doc_ids, raw_documents)))

    def get_retriever(self):
        return self.retriever


# LLM Selector to choose between ChatGroq and ChatOpenAI
class LLMSelector:
    def __init__(self):
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    def get_llm(self, llm_type: str):
        if llm_type == "groq":
            return ChatGroq(groq_api_key=self.GROQ_API_KEY, model_name="llama3-8b-8192")
        elif llm_type == "openai":
            return ChatOpenAI(model="gpt-4o-mini")
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")


# RAG Chain to manage the full RAG process and answer questions
class RAGChain:
    def __init__(self, retriever, llm_type: str = "groq"):
        self.retriever = retriever
        self.llm_selector = LLMSelector()
        self.llm = self.llm_selector.get_llm(llm_type)

    def create_rag_chain(self):
        # Define the system prompt
        system_prompt = """
        You are a travel assistant that helps people extract relevant information from documents 
        and assist in the travel questions. Based on the provided context, please answer the following question.

        Context: {context}

        Question: {input}

        Answer:
        """
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create the chain to pass the documents and the LLM to the prompt
        chain = create_stuff_documents_chain(self.llm, prompt)

        # Create the retrieval chain
        rag_chain = create_retrieval_chain(self.retriever, chain)
        return rag_chain

    def answer_question(self, question: str):
        rag_chain = self.create_rag_chain()
        result = rag_chain.invoke({"input": question})
        return result["answer"]


# Example usage
if __name__ == "__main__":
    # Load documents
    chunked_documents = load_docs_from_jsonl(
        "../data/processed_dataset/chunked_documents.jsonl"
    )
    raw_documents = load_docs_from_jsonl(
        "../data/processed_dataset/raw_documents.jsonl"
    )

    # Set up the vector store and ingest documents
    vector_store_manager = VectorStoreManager()
    vector_store_manager.ingest_documents(chunked_documents, raw_documents)

    # Set up the RAG chain
    retriever = vector_store_manager.get_retriever()
    rag_chain = RAGChain(retriever=retriever, llm_type="groq")

    # Answer a question
    question = "What is the cancellation policy for the aurora borealis tour?"
    answer = rag_chain.answer_question(question)
    print("Answer:", answer)
