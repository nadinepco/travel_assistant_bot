import json
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from langchain_community.document_loaders import PyPDFLoader
import pypdf
from langchain.schema import Document
from typing import Iterable
from dotenv import load_dotenv
import time
from langchain_openai import ChatOpenAI


class LLMTextChunker:
    def __init__(self, api_key: str, model_name="gpt-4o"):
        # def __init__(self, groq_api_key: str, model_name="llama3-70b-8192"):
        """
        Initialize the LLMTextChunker with the necessary API key and model name.
        """
        # self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
        self.llm = ChatOpenAI(api_key=api_key, model=model_name)

    def split_documents(self, document: Document) -> list:
        """
        Splits a LangChain Document into smaller chunks using the LLM.
        Args:
            document (Document): The input document to be chunked.

        Returns:
            List[Document]: A list of chunked documents with metadata.
        """
        # Define the prompt template for chunking the text
        prompt_template = """
        You're a professional editor highly skilled in data science. 
        You have been provided with a text input extracted from one or more PDF files that are related to travel bookings. 
        Your task is to analyze the content and organize it into related sections. 
        Determine the type of content that this has related to travel booking. 
        The sentences can be modified to make it understandable but don't remove any information.
        For each section, create an appropriate title that summarizes the content, using as many words from the original sentences as possible. 
        Ensure that each section is clearly separated by the delimiter '-~~~-'. 
        Only return the output format. Don't say anything else.
        Please follow these steps:

        - Determine the type of document. The type should only be one of the following: 'transportation','accommodation','activity'.
        - Identify groups of sentences or paragraphs that are closely related in topic or meaning.
        - Assign a concise, descriptive title to each group of related content.
        - Determine the location if mentioned and provide it for each section (e.g. Hotel, Country/Countries). The location should be the same for all sections.
        - If location is not provided, set "" in the location.
        - Don't remove important details like price, costs, cancellation fees, or any numbers
        - Ensure that sentences are readable and there is a space between words, and is understandable
        - Separate each section by the delimiter '-~~~-'.
        - For any named entity (e.g., people, places, organizations) that is accompanied by additional descriptive information, separate the descriptive information into its own distinct proposition.
        - Decontextualize propositions by adding necessary modifiers to nouns or sentences to ensure clarity, and replace any pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
        - Use only the provided information, don't add anything.
        - The output should be a string starting with the 'Title', 'Location', sentences. Each section separated by the delimiter.


        Title: <title>
        Location: <location>
        Type: <type>
        Sentences:<sentence 1> <sentence 2>...

        -~~~-

        Title: <title>
        Location: <location>
        Type: <type>
        Sentences: <sentence 1> <sentence 2>...

        -~~~-

        ```
        Input: 
        {input}
        ```
        """

        # Create the prompt template with LangChain
        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_template), ("human", "{input}")]
        )

        # Create the LLM chain and execute it
        chain = prompt | self.llm
        result = chain.invoke({"input": document.page_content})
        print("---Result---")
        print(result)
        # Use the LLM output to create chunked documents
        return self._convert_to_documents(result.content, document)

    def _convert_to_documents(
        self, chunked_text: str, original_document: Document
    ) -> list:
        """
        Converts the LLM output (chunked text) into LangChain Document objects.
        Args:
            chunked_text (str): The output from the LLM containing the chunked text.
            original_document (Document): The original document for reference metadata.

        Returns:
            List[Document]: A list of chunked documents with metadata.
        """
        # Parse the chunked text (assuming sections are split by '-~~~-')
        chunks = chunked_text.split("-~~~-")
        print("----Chunks----")
        print(chunks)
        documents = []
        for chunk_id, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                # Split chunk by the metadata indicators (Title and Location)
                title = self._extract_metadata(chunk, "Title")
                location = self._extract_metadata(chunk, "Location")
                doc_type = self._extract_metadata(chunk, "Type")
                sentences = self._extract_metadata(chunk, "Sentences")
                print(f"----Chunk {chunk_id}-----")
                print(f"Title: {title}")
                print(f"Location: {location}")
                print(f"Type: {doc_type}")
                print(f"Sentences: {sentences}")

                # Create a new Document with the extracted data
                chunked_document = Document(
                    page_content=f"{title}:{sentences}",
                    metadata={
                        "title": title,
                        "location": location,
                        "type": doc_type,
                        "source": original_document.metadata.get("source", ""),
                        "doc_id": original_document.metadata.get("doc_id", ""),
                        "chunk_id": chunk_id,
                    },
                )
                documents.append(chunked_document)

        return documents

    def _extract_metadata(self, chunk: str, key: str) -> str:
        """
        Extracts metadata like Title and Location from the chunked text.
        Args:
            chunk (str): The chunk of text.
            key (str): The metadata key to extract (e.g., "Title", "Location").

        Returns:
            str: The extracted metadata value.
        """
        for line in chunk.split("\n"):
            if line.startswith(f"{key}:"):
                return line[len(f"{key}:") :].strip()
        return ""

    def _get_filename(self, path: str) -> str:
        """
        Extract the base name from a file path without the extension.
        Args:
            path (str): The file path.

        Returns:
            str: The base file name without extension.
        """
        return os.path.splitext(os.path.basename(path))[0]


def save_docs_to_jsonl(array: Iterable[Document], file_path: str) -> None:
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            doc_dict = {
                "metadata": doc.metadata,
                "page_content": doc.page_content,
            }
            jsonl_file.write(json.dumps(doc_dict) + "\n")


def main():
    # Get files from folder
    pdf_folder_path = "data/"
    pdf_files = [
        os.path.join(pdf_folder_path, f)
        for f in os.listdir(pdf_folder_path)
        if f.endswith(".pdf")
    ]

    # Initialize the chunker
    # chunker = LLMTextChunker(groq_api_key=os.getenv("GROQ_API_KEY"))
    chunker = LLMTextChunker(api_key=os.getenv("OPENAI_API_KEY"))
    all_chunked_documents = []
    all_raw_documents = []
    # Preprocess the PDF files
    for id, file in enumerate(pdf_files):
        with pypdf.PdfReader(file) as pdf:
            full_text = ""
            # print the contents
            for page in pdf.pages:
                print(page.extract_text())
                full_text += page.extract_text()

        # create document for each file
        doc = Document(
            page_content=full_text,
            metadata={
                "source": file,
                "doc_id": id,
            },
        )
        all_raw_documents.append(doc)

        chunked_documents = chunker.split_documents(doc)
        all_chunked_documents.extend(chunked_documents)

        # Tokens for llama3-70b rate limit is 6k tokens per minute
        # Hence, I'll process each doc per minute to not hit the rate limit
        # time.sleep(60)

    save_docs_to_jsonl(
        all_chunked_documents,
        f"{pdf_folder_path}/processed_dataset/chunked_documents.jsonl",
    )
    save_docs_to_jsonl(
        all_raw_documents, f"{pdf_folder_path}/processed_dataset/raw_documents.jsonl"
    )


if __name__ == "__main__":
    main()
