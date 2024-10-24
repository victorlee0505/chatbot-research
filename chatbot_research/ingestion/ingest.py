import glob
import logging
import os
import sys
from multiprocessing import Pool
from typing import List

import chromadb
import openai
import pandas as pd
import torch
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    DataFrameLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredFileLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from chatbot_research.ingestion import ingest_code_text_splitter
from chatbot_research.ingestion.ingest_constants import (
    ALL_MINILM_L6_V2,
    CHROMA_SETTINGS_AZURE,
    CHROMA_SETTINGS_HF,
    PERSIST_DIRECTORY_AZURE,
    PERSIST_DIRECTORY_HF,
    STELLA_EN_1_5B_V5,
)
from chatbot_research.utils.git_repo_utils import EXTENSIONS, GitRepoUtils

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load environment variables
persist_directory_azure = PERSIST_DIRECTORY_AZURE
persist_directory_hf = PERSIST_DIRECTORY_HF
chroma_setting_azure = CHROMA_SETTINGS_AZURE
chroma_setting_hf = CHROMA_SETTINGS_HF
source_directory = "./source_documents"
chunk_size = 1000
chunk_overlap = 20


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
    ".log": (TextLoader, {"encoding": "unicode_escape"}),
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
        embedding_model: str = None,
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
        self.embedding_model = embedding_model
        self.failed_files = []

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
                    print(f"Loading csv: '{file_path}'")
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
                self.failed_files.append(file_path)
                return []
            else:
                try:
                    print(f"Loading file: '{file_path}'")
                    loader = loader_class(file_path, **loader_args)
                    return loader.load()
                except Exception as e:
                    print(e)
                    print(f"\nFile failed to load: '{file_path}'")
                    self.failed_files.append(file_path)
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

    def load_single_code(self, file_path: str) -> List[Document]:
        # print(f'load_single_code: {file_path}')
        ext = file_path.rsplit(".", 1)[-1]
        if ext not in EXTENSIONS:
            print(f"Skipping extension: '{ext}")
            return []
        try:
            with open(file_path, encoding="utf-8") as f:
                texts = ingest_code_text_splitter.codeTextSplitter(
                    language=ingest_code_text_splitter.get_language(ext),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    documents=f.read(),
                )
                f.close
                return texts
        except:
            print(f"\nFile failed to load: '{file_path}', File Types: '{ext}'")
            return []

    def process_code(self) -> List[Document]:
        """
        Load Code and split in chunks
        """
        # Find all Git repositories within the provided repo_path and its subfolders
        git_repo_utils = GitRepoUtils(source_path=self.source_path)
        all_files = git_repo_utils.find_all_files()
        print(f"Total files found: {len(all_files)}")

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(
                total=len(all_files), desc="Loading new documents", ncols=80
            ) as pbar:
                for i, docs in enumerate(
                    pool.imap_unordered(self.load_single_code, all_files)
                ):
                    results.extend(docs)
                    pbar.update()
        print(f"Split into {len(results)} chunks of text (max. {chunk_size} char each)")
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
        # if self.offline:
        #     chunk_size = 1000
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} char each)")
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
        print(f"Offline Embedding: {self.offline}")
        torch.set_num_threads(os.cpu_count())
        # Create embeddings
        if self.offline:
            if not self.gpu:
                print("Disable CUDA")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                torch.device("cpu")
            if self.embedding_model is None:
                embeddings = HuggingFaceEmbeddings()
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={"trust_remote_code": True},
                )
            self.chroma_setting = (
                CHROMA_SETTINGS_HF
                if self.chroma_setting is None
                else self.chroma_setting
            )
            self.persist_directory = (
                persist_directory_hf
                if self.persist_directory is None
                else self.persist_directory
            )
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
            self.persist_directory = persist_directory_azure

        client = chromadb.PersistentClient(
            settings=self.chroma_setting, path=self.persist_directory
        )
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
            texts = self.process_documents(
                [metadata["source"] for metadata in collection["metadatas"]]
            )
            print(f"Failed Documents: {self.failed_files}")
            # texts.append(self.process_code())
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            print("Documents: Creating new vectorstore")
            texts = self.process_documents()
            print(f"Failed Documents: {self.failed_files}")
            # texts.append(self.process_code())
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db = Chroma.from_documents(
                    client=client,
                    documents=texts,
                    embedding=embeddings,
                )
        db = None
        print(f"Documents: Ingestion complete!")

        if self.does_vectorstore_exist(self.persist_directory):
            # Update and store locally vectorstore
            print(
                f"Code: Appending to existing vectorstore at {self.persist_directory}"
            )
            db = Chroma(
                client=client,
                embedding_function=embeddings,
            )
            collection = db.get()
            texts = self.process_code()
            if texts:
                print(f"Code: Creating embeddings. May take some minutes...")
                db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            print("Code: Creating new vectorstore")
            texts = self.process_code()
            if texts:
                print(f"Code: Creating embeddings. May take some minutes...")
                db = Chroma.from_documents(
                    client=client,
                    documents=texts,
                    embedding=embeddings,
                )
        db = None
        print(f"Code: Ingestion complete!")
