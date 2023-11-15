from langchain.retrievers import ParentDocumentRetriever
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from chatbot_research.ingestion.file_system import LocalFileStore
import chromadb
from chatbot_research.ingestion.ingest_constants import CHROMA_SETTINGS_HF, PERSIST_DIRECTORY_HF, PERSIST_DIRECTORY_PARENT_HF

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

model_kwargs = {'device': 'cpu'}
embedding_llm = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

client = chromadb.PersistentClient(settings=CHROMA_SETTINGS_HF, path=PERSIST_DIRECTORY_HF)
local_store = LocalFileStore(PERSIST_DIRECTORY_PARENT_HF)
db = Chroma(
    client=client,
    embedding_function=embedding_llm,
)
# retriever = db.as_retriever(
#     search_type="similarity", search_kwargs={"k": self.llm_config.target_source_chunks}, max_tokens_limit=self.llm_config.retriever_max_tokens_limit
# )
local_store = LocalFileStore(PERSIST_DIRECTORY_PARENT_HF)
retriever = ParentDocumentRetriever(
    vectorstore=db, 
    docstore=local_store, 
    child_splitter=child_splitter,
)

sub_docs = db.similarity_search(query="spring", k=20)

for i, doc in sub_docs:
    print(f"Child {i}: {doc}")
    print()


retrieved_docs = retriever.get_relevant_documents(query="spring", top_k=20)
for i, doc in retrieved_docs:
    print(f"Parent {i}: {doc}")
    print()