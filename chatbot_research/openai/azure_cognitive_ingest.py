import glob
import os
from multiprocessing import Pool
from typing import List

import openai
import pandas as pd
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader,
    DataFrameLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm

from azure_cognitive_search import AzureCognitiveSearch

from chatbot_research.ingestion.ingest_constants import CHROMA_SETTINGS_AZURE, PERSIST_DIRECTORY_AZURE

openai.api_type = "azure"
openai.api_base = os.getenv(
    "AZURE_OPENAI_BASE_URL"
)  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # this may change in the future
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = "text-embedding-ada-002"

vector_store_address = os.getenv("AZURE_OPENAI_SEARCH_ENDPOINT")
vector_store_password = os.getenv("AZURE_OPENAI_SEARCH_ADMIN_KEY")

source_directory = "./source_documents"
chunk_size = 1000
chunk_overlap = 0


class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf8"}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {"encoding": "utf8"}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


class AzureCognitiveSearchIngestion:
    # initialize
    def __init__(
        self,
        index_name: str = None,
        source_path: str = None,
    ):
        """
        Initialize Ingestion object

        Parameters:
        - source_path: default is "./source_documents"
        """
        self.index_name = index_name
        self.source_path = source_path
        self.persist_directory = PERSIST_DIRECTORY_AZURE
        self.chroma_setting = CHROMA_SETTINGS_AZURE
        if self.index_name is None or len(self.index_name) == 0:
            raise ValueError("index_name can not be null.")
        if self.source_path is None or len(self.source_path) == 0:
            self.source_path = source_directory
        self.main()

    def load_single_document(self, file_path: str) -> List[Document]:
        # print(f'load_single_document: {file_path}')
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            if ext == ".csv":
                try:
                    loader = loader_class(file_path, **loader_args)
                    return loader.load()
                except:
                    pass
                try:
                    df = pd.read_csv(file_path)
                    loader = DataFrameLoader(df)
                    return loader.load()
                except:
                    pass
                print(f"\ncsv failed to load: '{file_path}'")
                return []
            else:
                try:
                    loader = loader_class(file_path, **loader_args)
                    return loader.load()
                except:
                    print(f"\nFile failed to load: '{file_path}'")
                    return []
        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(
        self, source_dir: str, ignored_files: List[str] = []
    ) -> List[Document]:
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        all_files = []
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        filtered_files = [
            file_path for file_path in all_files if file_path not in ignored_files
        ]

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(
                total=len(filtered_files), desc="Loading new documents", ncols=80
            ) as pbar:
                for i, docs in enumerate(
                    pool.imap_unordered(self.load_single_document, filtered_files)
                ):
                    results.extend(docs)
                    pbar.update()
        return results

    def process_documents(self, ignored_files: List[str] = []) -> List[Document]:
        """
        Load documents and split in chunks
        """
        print(f"Loading documents from {self.source_path}")
        documents = self.load_documents(self.source_path, ignored_files)
        if not documents:
            print("No new documents to load")
            return None
        print(f"Loaded {len(documents)} new documents from {self.source_path}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
        return texts

    def does_vectorstore_exist(self, persist_directory: str) -> bool:
        """
        Checks if vectorstore exists
        """
        if os.path.exists(os.path.join(persist_directory, "index")):
            if os.path.exists(
                os.path.join(persist_directory, "chroma-collections.parquet")
            ) and os.path.exists(
                os.path.join(persist_directory, "chroma-embeddings.parquet")
            ):
                list_index_files = glob.glob(
                    os.path.join(persist_directory, "index/*.bin")
                )
                list_index_files += glob.glob(
                    os.path.join(persist_directory, "index/*.pkl")
                )
                # At least 3 documents are needed in a working vectorstore
                if len(list_index_files) > 3:
                    return True
        return False

    def main(self):
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            deployment="text-embedding-ada-002",
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type=openai.api_type,
            openai_api_version=openai.api_version,
            chunk_size=1,
        )

        # Create Index (update if already exist)
        azuredb = AzureCognitiveSearch(
            azure_search_endpoint=vector_store_address,
            azure_search_key=vector_store_password,
            index_name=index_name,
            embedding_function=embeddings.embed_query,
            # search_type='semantic_hybrid', # need to config
            semantic_configuration_name="my_semantic_config",
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type=openai.api_type,
            openai_api_version=openai.api_version,
        )

        if self.does_vectorstore_exist(self.persist_directory):
            # Update and store locally vectorstore and Azure Cognitive Index
            print(
                f"Documents: Appending to existing vectorstore at {self.persist_directory}"
            )
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                client_settings=self.chroma_setting,
            )
            collection = db.get()
            texts = self.process_documents(
                [metadata["source"] for metadata in collection["metadatas"]]
            )
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db.add_documents(texts)
                azuredb.add_documents(documents=texts)
        else:
            # Create and store locally vectorstore and Azure Cognitive Index
            print("Documents: Creating new vectorstore")
            texts = self.process_documents()
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db = Chroma.from_documents(
                    texts,
                    embeddings,
                    persist_directory=self.persist_directory,
                    client_settings=self.chroma_setting,
                )
                azuredb.add_documents(documents=texts)
        db.persist()
        db = None
        print(f"Documents: Ingestion complete!")

        # if self.does_vectorstore_exist(self.persist_directory):
        #     # Update and store locally vectorstore
        #     print(
        #         f"Code: Appending to existing vectorstore at {self.persist_directory}"
        #     )
        #     db = Chroma(
        #         persist_directory=self.persist_directory,
        #         embedding_function=embeddings,
        #         client_settings=self.chroma_setting,
        #     )
        #     collection = db.get()
        #     texts = self.process_code()
        #     if texts:
        #         print(f"Code: Creating embeddings. May take some minutes...")
        #         db.add_documents(texts)
        # else:
        #     # Create and store locally vectorstore
        #     print("Code: Creating new vectorstore")
        #     texts = self.process_code()
        #     if texts:
        #         print(f"Code: Creating embeddings. May take some minutes...")
        #         db = Chroma.from_documents(
        #             texts,
        #             embeddings,
        #             persist_directory=self.persist_directory,
        #             client_settings=self.chroma_setting,
        #         )
        # db.persist()
        # db = None
        # print(f"Code: Ingestion complete!")


if __name__ == "__main__":
    index_name: str = "langchain-eng-demo"

    # overiding default source path
    base_path = "C:\\Users\\LeeVic\\workspace\\openai\\chatbot-research\\temp"

    # Offline
    # ingest = Ingestion(offline=True, source_path=base_path)
    # ingest = Ingestion(offline=True, gpu=True, source_path=base_path)

    # Azure Open AI
    # ingest = Ingestion(offline=False)

    # Azure Open AI
    # ingest = AzureCognitiveSearchIngestion(index_name=index_name)
    ingest = AzureCognitiveSearchIngestion(index_name=index_name, source_path=base_path)
