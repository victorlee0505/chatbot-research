import os
import pprint

import chromadb
import chromadb.config
import torch
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from chatbot_research.ingestion.ingest import Ingestion
from chatbot_research.ingestion.ingest_constants import (
    CHROMA_SETTINGS_HF,
    PERSIST_DIRECTORY_HF,
    STELLA_EN_1_5B_V5,
)
from chatbot_research.ingestion.ingest_faiss import IngestionFAISS

persist_directory = PERSIST_DIRECTORY_HF
offline = True
k = 5


def main():
    # chromadb.config.logger.setLevel(level="DEBUG")
    # ingest = Ingestion(offline=offline, embedding_model=STELLA_EN_1_5B_V5)
    ingest = IngestionFAISS(offline=offline)
    db = ingest.run_ingest()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.device("cpu")

    retriever: VectorStoreRetriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
        max_tokens_limit=1000,
    )

    stop = False

    while not stop:
        query = input("Enter your query: ")
        if query == "bye":
            stop = True
        else:
            results = retriever.invoke(query)
            i = 0
            if len(results) == 0:
                print("No results found")
            else:
                for result in results:
                    print(f"\nindex {i}:")
                    pprint.pprint(result.model_dump())
                    i = i + 1


if __name__ == "__main__":
    main()
