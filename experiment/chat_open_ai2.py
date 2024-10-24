# import os
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# from llama_index import GPTVectorStoreIndex, GPTEmptyIndex, GPTTreeIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext
# from llama_index.indices.composability import ComposableGraph
# from langchain.chat_models import ChatOpenAI

# # os.environ['OPENAI_API_KEY'] = "YOUR API KEY"

# index_summary = "This document describes Paul Graham's life, from early adulthood to the present day."
# empty_index_summary = "This can be used for general knowledge purposes."

# documents = SimpleDirectoryReader('./data/doc').load_data()

# llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)
# storage_context = StorageContext.from_defaults()

# index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
# empty_index = GPTEmptyIndex(service_context=service_context, storage_context=storage_context)

# custom_query_engines = {
#     index.index_id: index.as_query_engine(
#         similarity_top_k=3,
#         response_mode="tree_summarize",
#     )
# }

# graph2 = ComposableGraph.from_indices(
#     GPTTreeIndex,
#     [index, empty_index],
#     index_summaries=[index_summary, empty_index_summary]
# )

# query_engine = graph2.as_query_engine(
#     custom_query_engines=custom_query_engines
# )

# response = query_engine.query(
#     "What did the author do growing up?", 
# )

# print(str(response))

# # response.print_response_stream()

# response2 = query_engine.query(
#     "Who is Obama?", 
# )
# print(str(response2))