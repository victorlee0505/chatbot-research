from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/doc/paul_graham_essay.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(documents=docs, embedding=embeddings)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2}, max_tokens_limit=1000)

query = "What did the author do growing up?"

results = retriever.get_relevant_documents(query)

i = 0
for result in results:
    print(f'index {i}: {result}')
    i=i+1

