import json
import os

import openai
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # this may change in the future
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = "text-embedding-ada-002"

vector_store_address = os.getenv("AZURE_OPENAI_SEARCH_ENDPOINT")
vector_store_password = os.getenv("AZURE_OPENAI_SEARCH_ADMIN_KEY")

index_name: str = "langchain-vector-demo4"

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
    chunk_size=1,
)
db = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    # search_type='semantic_hybrid', # need to config
    semantic_configuration_name='my_semantic_config',
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
)

df = pd.read_json('./data/doc/text-sample.json')
print(df)
loader = DataFrameLoader(data_frame=df, page_content_column="title")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

db.add_documents(documents=docs) # this will upload a copy of the documents with the index name (so do it once then you can re-use the index)
