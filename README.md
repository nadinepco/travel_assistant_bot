# Personal Travel Assistant

## Problem Description
When planning a trip, travelers often have multiple documents spread across different platforms, such as hotel bookings, flight confirmations, and activity reservations, often in PDF format. Managing this information manually can be cumbersome, leading to confusion or missed details. Travelers frequently need quick access to specific booking information, such as accommodation addresses, check-in times, or flight details, and navigating through multiple files or platforms to retrieve this information is time-consuming.

This project solves the problem of efficiently accessing travel-related information. The Travel Assistant is a RAG-based application that allows users to interact with a large language model (LLM) and retrieve specific details from their travel documents using natural language queries. Instead of searching through emails or PDFs, users can simply directly ask questions about their trip.

## Dataset
For this project, dummy travel files in PDF format are used to simulate real travel documents.
The data used for this project includes various travel-related documents in PDF format, such as:
- Accommodation bookings
- Transportation details (flight, train, bus bookings)
- Booked activities and tours

### Chunking the Dataset
This was the tricky part as PDF files, especially with travel documents are not well structured. This took most of my time to figure out. I had tried RecursiveTextSplitter and Semantic Chunking but I ended up with using LLM-based Thematic chunking.
These documents are then preprocessed as follows: 
- Read each document and pass it to the LLM to analyze the content and organize it to related sections
- Each related section is one chunk 
- The results are converted and exported to a jsonl 
- When using this approach, it's important to account for the size of the PDF documents to avoid surpassing token constraints during processing (rate limits for Groq models)

You can find the data in [data](data) and the processed_data(chunks) in [data/procesed_data](data/procesed_data)

## Technologies
- Python 3.11
- [Open AI](https://openai.com/) and [Groq](https://groq.com/) for LLMs
- [LangChain](https://langchain.readthedocs.io/en/latest/index.html): Framework for managing and chaining language models and prompts
- [Streamlit](https://streamlit.io/): For creating the web interface of the chatbot
- [Chroma](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/): For vectorestore and search

## Local Setup
1. Prerequisites
    - Python 3.11+
    - [OpenAI API Key](https://platform.openai.com/api-keys)
    - [Groq API Key](https://console.groq.com/keys)
2. Clone the repository <br/>
    `git clone https://github.com/nadinepco/travel_assistant_bot.git`
3. Create a new environment. I have used conda.<br/>
    `conda create --name <env_name>`

3. Install the requirements file in a new environment <br/>
    `pip install -r requirements.txt`
4. Create a .env file in the root directory and add your API key:
    ```
    OPENAI_API_KEY=your_openai_api_key 
    GROQ_API_KEY=your_groq_api_key
    ```
5. Start the streamlit app. <br/>
    `streamlit run app.py`

## Docker Setup
To run the Travel Assistant using Docker, follow these steps:

1. **Build the Docker Image**:
    ```bash
    docker build -t travel_assistant_bot .
    ```

2. **Run the Docker Container**:
    ```bash
    docker run -p 8501:8501 travel_assistant_bot streamlit run app.py
    ```

This will build and start the Streamlit application inside a Docker container. You can access the app by navigating to `http://localhost:8501` in your browser.

## Evaluation
The code for evaluating the Retrieval and RAG flow can be found in this [notebook](notebooks/evaluation.ipynb)

### Retrieval
The `similarity_search` function using cosine similarity against the ground truth data was evaluated. 
The evaluation metrics used were:

- **Hit Rate**: The proportion of queries where the relevant document or document chunk was retrieved within the top `k` results.
- **Mean Reciprocal Rank (MRR)**: The average reciprocal rank of the first relevant document or document chunk across all queries.

Given the multi-document nature of the dataset, the retrieval performance at both the document and chunk levels using doc_id and chunk_id identifiers was evaluated:

- **Document IDs (`doc_id`) only**: Evaluating whether the correct document was retrieved, regardless of the chunk.
- **Document IDs and Chunk IDs (`doc_id` and `chunk_id`)**: Evaluating whether the exact relevant chunk was retrieved.

### RAG Flow and RAG Evaluation
The RAG was evaluated using LLM-as-a-Judge (with gpt-4o) for checking if the answer is relevant, partially relevant or non-relevant.
Admittedly, this is a metric I shouldn't consider as all the files are related to travel bookings, hence the high result as relevant.
- Relevant           96%
- Partly_relevant    3%
- Non_relevant       1%

### Resources:
* [5 Levels of Text Splitting for Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4&t=2882s)