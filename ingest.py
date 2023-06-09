import glob
import logging
import os
import sys
from multiprocessing import Pool
from typing import List

import openai
import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    DataFrameLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

import code_text_splitter
from constants import CHROMA_SETTINGS
from git_repo_utils import EXTENSIONS, GitRepoUtils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_AZURE_BASE_URL")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables
persist_directory = "./storage"
source_directory = "./source_documents"
chunk_size = 500
chunk_overlap = 50

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
    # ".enex": (EverNoteLoader, {}),
    # ".eml": (MyElmLoader, {}),
    # ".epub": (UnstructuredEPubLoader, {}),
    # ".html": (UnstructuredHTMLLoader, {"encoding": "utf8"}),
    # ".md": (UnstructuredMarkdownLoader, {}),
    # ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
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
        source_path: str = None,
        gpu: bool = False,
    ):
        """
        Initialize AzureOpenAiChatBot object

        Parameters:
        - source_path: optional
        - open_chat: set True to allow answer outside of the context
        - load_data: set True if you want to load new / additional data. default skipping ingest data.
        - show_stream: show_stream
        - show_source: set True will show source of the completion
        """
        self.offline = offline
        self.source_path = source_path
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
                except:
                    pass
                try:
                    df = pd.read_csv(file_path)
                    loader = DataFrameLoader(df)
                except:
                    pass
                try:
                    return loader.load()
                except:
                    print(f"csv failed to load: '{file_path}'")
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

    def load_single_code(self, file_path: str) -> List[Document]:
        # print(f'load_single_code: {file_path}')
        ext = file_path.rsplit(".", 1)[-1]
        if ext not in EXTENSIONS:
            print(f"Skipping extension: '{ext}")
            return []
        try:
            with open(file_path, encoding="utf-8") as f:
                texts = code_text_splitter.codeTextSplitter(
                    language=code_text_splitter.get_language(ext),
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
        print(
            f"Split into {len(results)} chunks of text (max. {chunk_size} tokens each)"
        )
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
        return documents

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
        # Create embeddings
        if self.offline:
            if not self.gpu:
                print("Disable CUDA")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            embeddings = HuggingFaceEmbeddings()
        else:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                deployment="text-embedding-ada-002",
                openai_api_key=openai.api_key,
                openai_api_base=openai.api_base,
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
                chunk_size=1,
            )

        if self.does_vectorstore_exist(persist_directory):
            # Update and store locally vectorstore
            print(
                f"Documents: Appending to existing vectorstore at {persist_directory}"
            )
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS,
            )
            collection = db.get()
            texts = self.process_documents(
                [metadata["source"] for metadata in collection["metadatas"]]
            )
            # texts.append(self.process_code())
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db.add_documents(texts)
        else:
            # Create and store locally vectorstore
            print("Documents: Creating new vectorstore")
            texts = self.process_documents()
            # texts.append(self.process_code())
            if texts:
                print(f"Documents: Creating embeddings. May take some minutes...")
                db = Chroma.from_documents(
                    texts,
                    embeddings,
                    persist_directory=persist_directory,
                    client_settings=CHROMA_SETTINGS,
                )
        db.persist()
        db = None
        print(f"Documents: Ingestion complete!")

        if self.does_vectorstore_exist(persist_directory):
            # Update and store locally vectorstore
            print(f"Code: Appending to existing vectorstore at {persist_directory}")
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS,
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
                    texts,
                    embeddings,
                    persist_directory=persist_directory,
                    client_settings=CHROMA_SETTINGS,
                )
        db.persist()
        db = None
        print(f"Code: Ingestion complete!")

if __name__ == "__main__":
    base_path = "C:\\Users\\victo\\workspaces\\OpenAI"
    ingest = Ingestion(offline=True, gpu=True, source_path=base_path)
