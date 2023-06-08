import os
import json
import openai
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    GPTVectorStoreIndex,
    GPTEmptyIndex,
    GPTTreeIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
)
from llama_index.indices.composability import ComposableGraph
import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_type = "azure"
openai.api_base = "https://cog-frutomt5wmbzu.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")
deployment_name = "chat"

llm = AzureOpenAI(
    deployment_name=deployment_name,
    model_kwargs={
        "api_key": openai.api_key,
        "api_base": openai.api_base,
        "api_type": openai.api_type,
        "api_version": openai.api_version,
    },
)
llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="text-embedding-ada-002",
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)

documents = SimpleDirectoryReader("./data/doc").load_data()

# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 50
# set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, embed_model=embedding_llm, prompt_helper=prompt_helper
)
storage_context = StorageContext.from_defaults()

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
empty_index = GPTEmptyIndex(
    service_context=service_context, storage_context=storage_context
)

custom_query_engines = {
    index.index_id: index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
    )
}

graph2 = ComposableGraph.from_indices(
    GPTTreeIndex,
    [index, empty_index],
    index_summaries=["index_summary", "empty_index_summary"],
)

query_engine = graph2.as_query_engine(custom_query_engines=custom_query_engines)

query = "What did the author do growing up?"
answer = query_engine.query(query)

print("query was:", query)
print("answer was:", answer)

query = "What is most interesting about this essay?"
answer = query_engine.query(query)

print("query was:", query)
print("answer was:", answer)

query = "Who is Obama?"
answer = query_engine.query(query)

print("query was:", query)
print("answer was:", answer)

# print(answer.get_formatted_sources())
