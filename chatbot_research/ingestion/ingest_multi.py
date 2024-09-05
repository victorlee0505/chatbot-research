from dotenv import load_dotenv
import glob
import logging
import os
import sys
from multiprocessing import Pool
from typing import List
import chromadb
import uuid
import pickle

import openai
import pandas as pd
import torch
from chromadb.config import Settings
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
from chatbot_research.ingestion.file_system import LocalFileStore
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm

from chatbot_research.ingestion import ingest_code_text_splitter
from chatbot_research.ingestion.ingest_constants import (
    CHROMA_SETTINGS_AZURE,
    CHROMA_SETTINGS_HF,
    CHROMA_SETTINGS_PARENT_HF,
    PERSIST_DIRECTORY_AZURE,
    PERSIST_DIRECTORY_HF,
    PERSIST_DIRECTORY_PARENT_HF,
)
from chatbot_research.utils.git_repo_utils import EXTENSIONS, GitRepoUtils

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load environment variables
source_directory = "./source_documents"
CHUNK_SIZE_PARENT = 2000
CHUNK_SIZE = 400
chunk_overlap = 20

id_key: str = "doc_id"

# Custom document loaders
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


class Ingestion:
    # initialize
    def __init__(
        self,
        offline: bool = False,
        openai: bool = False,
        source_path: str = None,
        chroma_setting: Settings = None,
        persist_directory: str = None,
        gpu: bool = False,
    ):
        """
        Initialize Ingestion object

        Parameters:
        - offline: embedding engine: offline= huggingface , online = azure. they are not cross compatible.
        - source_path: default is "./source_documents"
        - chroma_setting: vector storage setting, online and offline has different setting (recommend keep it default)
        - persist_directory: vector storage location, online and offline has different storage (recommend keep it default)
        - gpu: enable CUDA if supported. no effect to online embedding
        """
        self.offline = offline
        self.openai = openai
        self.source_path = source_path
        self.chroma_setting = chroma_setting
        self.persist_directory = persist_directory
        self.gpu = gpu

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
                except Exception as e:
                    print(e)
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

    def process_documents(self, ignored_files: List[str] = []):
        """
        Load documents and split in chunks
        """
        print(f"Loading documents from {self.source_path}")
        documents = self.load_documents(self.source_path, ignored_files)
        if not documents:
            print("No new documents to load")
            return None
        print(f"Loaded {len(documents)} new documents from {self.source_path}")
        # if self.offline:
        #     chunk_size = 1000
        text_splitter_parent = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_PARENT, chunk_overlap=chunk_overlap
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=chunk_overlap
        )
        full_documents = text_splitter_parent.split_documents(documents)
        doc_ids = [str(uuid.uuid4()) for _ in full_documents]
        docs = []
        full_docs = []
        for i, doc in enumerate(full_documents):
            _id = doc_ids[i]
            sub_docs = text_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        print(f"Split into child {len(docs)} chunks of text (max. {CHUNK_SIZE} char each)")
        print(f"Split into parent {len(full_docs)} chunks of text (max. {CHUNK_SIZE_PARENT} char each)")
        return docs, full_docs

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
        print(f"Offline Embedding: {self.offline}")
        torch.set_num_threads(os.cpu_count())
        # Create embeddings
        if self.offline:
            if not self.gpu:
                print("Disable CUDA")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                torch.device('cpu')
            embeddings = HuggingFaceEmbeddings()
            self.chroma_setting = CHROMA_SETTINGS_HF
            self.chroma_setting_parent = CHROMA_SETTINGS_PARENT_HF
            self.persist_directory = PERSIST_DIRECTORY_HF
            self.persist_directory_parent = PERSIST_DIRECTORY_PARENT_HF
        else:
            if not self.openai:
                openai.api_type = "azure"
                openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
                openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")
                openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
                
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    deployment="text-embedding-ada-002",
                    openai_api_key=openai.api_key,
                    openai_api_base=openai.api_base,
                    openai_api_type=openai.api_type,
                    openai_api_version=openai.api_version,
                    chunk_size=1,
                )
            else:
                print("OpenAI Embedding")
                openai.api_key = os.getenv("OPENAI_API_KEY")
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=openai.api_key,
                )
            self.chroma_setting = CHROMA_SETTINGS_AZURE
            self.persist_directory = PERSIST_DIRECTORY_AZURE

        client = chromadb.PersistentClient(settings=self.chroma_setting, path=self.persist_directory)
        local_store = LocalFileStore(self.persist_directory_parent)
        if self.does_vectorstore_exist(self.persist_directory):
            # Update and store locally vectorstore
            print(
                f"Documents: Appending to existing vectorstore at {self.persist_directory}"
            )
            db = Chroma(
                client=client,
                embedding_function=embeddings,
            )
            collection = db.get()
            texts, full_texts = self.process_documents(
                [metadata["source"] for metadata in collection["metadatas"]]
            )
            # texts.append(self.process_code())
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db.add_documents(texts)
                local_store.mset(full_texts)
        else:
            # Create and store locally vectorstore
            print("Documents: Creating new vectorstore")
            texts, full_texts = self.process_documents()
            print(f"texts: {texts.pop()}")
            print(f"full_texts: {full_texts.pop()}")
            # texts.append(self.process_code())
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db = Chroma.from_documents(
                    client=client,
                    documents=texts,
                    embedding=embeddings,
                )
                local_store.mset(full_texts)
                    
        db = None
        print(f"Documents: Ingestion complete!")


if __name__ == "__main__":

    # overiding default source path
    # base_path = "C:\\path\\to\\your\\data"
    # base_path = "C:\\Users\\LeeVic\\workspace\\openai\\chatbot-research\\temp"

    # Offline
    ingest = Ingestion(offline=True, gpu=True)
    # ingest = Ingestion(offline=True, source_path=base_path)
    # ingest = Ingestion(offline=True, gpu=True, source_path=base_path)

    # Azure Open AI
    # ingest = Ingestion(offline=False)
    # ingest = Ingestion(offline=False, source_path=base_path)

    # OpenAI
    # ingest = Ingestion(offline=False, openai=True)

