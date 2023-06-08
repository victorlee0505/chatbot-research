import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI

# os.environ['OPENAI_API_KEY'] = "YOUR API KEY"

documents = SimpleDirectoryReader('./data/doc').load_data()

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine(
    service_context=service_context,
    similarity_top_k=3,
    streaming=True,
)
response = query_engine.query(
    "What did the author do growing up?", 
)
response.print_response_stream()