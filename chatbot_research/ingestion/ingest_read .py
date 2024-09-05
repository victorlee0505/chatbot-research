from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from file_system import LocalFileStore
import chromadb
from chatbot_research.ingestion.ingest_constants import CHROMA_SETTINGS_HF, PERSIST_DIRECTORY_HF, PERSIST_DIRECTORY_PARENT_HF

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

model_kwargs = {'device': 'cpu'}
embedding_llm = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

client = chromadb.PersistentClient(settings=CHROMA_SETTINGS_HF, path=PERSIST_DIRECTORY_HF)

db = Chroma(
    client=client,
    embedding_function=embedding_llm,
)

sub_docs = db.similarity_search(query="angular", k=20)

for i, doc in sub_docs:
    print(f"Child {i}: {doc}")
    print()
