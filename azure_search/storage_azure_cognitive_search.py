import json
import os

import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever
from langchain.retrievers import AzureCognitiveSearchRetriever

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # this may change in the future
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = "text-embedding-ada-002"

# vector_store_address = os.getenv("AZURE_OPENAI_SEARCH_ENDPOINT")
vector_store_password = os.getenv("AZURE_OPENAI_SEARCH_ADMIN_KEY")

vector_store_service_name = os.getenv("AZURE_OPENAI_SERVICE_NAME")

index_name = "langchain-vector-demo4"

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment="text-embedding-ada-002",
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_type=openai.api_type,
    openai_api_version=openai.api_version,
    chunk_size=1,
)
# db = AzureSearch(
#     azure_search_endpoint=vector_store_address,
#     azure_search_key=vector_store_password,
#     index_name=index_name,
#     embedding_function=embeddings.embed_query,
#     # search_type='semantic_hybrid', # need to config
#     semantic_configuration_name='my_semantic_config',
#     openai_api_key=openai.api_key,
#     openai_api_base=openai.api_base,
#     openai_api_type=openai.api_type,
#     openai_api_version=openai.api_version,
# )

# df = pd.read_json('./data/doc/text-sample.json')
# print(df)
# loader = DataFrameLoader(data_frame=df, page_content_column="title")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
# docs = text_splitter.split_documents(documents)

# db.add_documents(documents=docs) # this will upload a copy of the documents with the index name (so do it once then you can re-use the index)

# # this will return VectorStoreRetriever which does not have Cognitive search
retriever = AzureCognitiveSearchRetriever(
    service_name = vector_store_service_name,
    index_name = index_name,
    api_key = vector_store_password,
    api_version = openai.api_version,
    content_key =  "content",
    )


# retriever.service_name = vector_store_service_name
# retriever.index_name = index_name
# retriever.api_key = vector_store_password
# retriever.api_version = openai.api_version
# retriever.content_key =  "content"
    

# this retriever is correct but useless as it does not pass 'filter'
# https://github.com/hwchase17/langchain/issues/6131 hinted to them hope they do a PR or i will do PR
# retriever = AzureSearchVectorStoreRetriever(vectorstore=db)

# print(f'retriever search_type: {retriever.search_type}')
# print(f'retriever2 search_type: {retriever2.search_type}')


query = "tools for software development" 
# query = "solution de stockage évolutive"

# retriever.search_kwargs = {'filters': "category eq 'Developer Tools'"}
results1 = retriever.get_relevant_documents(query)

results2 = retriever.get_relevant_documents(query)

results3 = retriever.get_relevant_documents(query)

# print(f'search_type1: {retriever.search_type}')
# i = 0
# for result in results:
#     print(f"index {i}: {result}")
#     i = i + 1

# print(f'search_type2: {retriever2.search_type}')

# use the vector store instead as retriever is useless which does not pass search_kwargs
# results1 = retriever.get_relevant_documents(query, k=3, kwargs={'filters': "category eq 'Developer Tools'"})

# results2 = retriever.hybrid_search(query, k=3, kwargs={'filters': "category eq 'Developer Tools'"})

# results3 = retriever.semantic_hybrid_search(query, k=3, kwargs={'filters': "category eq 'Developer Tools'"})


print('\nsimilarity_search\n')
i = 0
for result in results1:
    metadata = result.metadata
    print(f"index {i}:")
    print(f"Result: {result.page_content}; ID: {metadata['id']}; category: {metadata['category']}")
    i = i + 1

print('\nhybrid_search\n')
i = 0
for result in results2:
    metadata = result.metadata
    print(f"index {i}:")
    print(f"Result: {result.page_content}; ID: {metadata['id']}; category: {metadata['category']}")
    i = i + 1

print('\nsemantic_hybrid_search\n')
i = 0
for result in results3:
    metadata = result.metadata
    print(f"index {i}:")
    print(f"Result: {result.page_content}; ID: {metadata['id']}; category: {metadata['category']}")
    i = i + 1
