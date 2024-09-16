import os
from dotenv import load_dotenv
from typing import Iterable
from langchain_core.documents import Document
import json


def get_api_key(key_name):
    """Retrieve an API key from environment variables.

    Args:
        key_name (str): The name of the environment variable to fetch.

    Returns:
        str: The API key associated with the given key name.

    Raises:
        ValueError: If the API key is not found.
    """
    load_dotenv()
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(
            f"API key for {key_name} not found. Please set the {key_name} environment variable."
        )
    return api_key


def load_docs_from_jsonl(file_path: str) -> Iterable[Document]:
    array = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
