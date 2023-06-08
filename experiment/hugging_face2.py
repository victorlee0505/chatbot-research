import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import torch
from llama_index import GPTVectorStoreIndex, GPTEmptyIndex, GPTTreeIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext
from llama_index.indices.composability import ComposableGraph
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llm_predictor import HuggingFaceLLMPredictor
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding

# comment this to use CUDA
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

documents = SimpleDirectoryReader('./data/doc').load_data()

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""" 

index_summary = "This document describes Paul Graham's life, from early adulthood to the present day."
empty_index_summary = "This can be used for general knowledge purposes."

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<human>:{query_str}<bot>:")

hf_predictor = HuggingFaceLLMPredictor(
    max_input_size=4096, 
    max_new_tokens=256,
    temperature=0.7,
    do_sample=False,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"offload_folder": "offload"}
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"offload_folder": "offload", "torch_dtype": torch.float16}
)

service_context = ServiceContext.from_defaults(llm_predictor=hf_predictor, chunk_size_limit=1024)

storage_context = StorageContext.from_defaults()

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, embed_model=embed_model)
empty_index = GPTEmptyIndex(service_context=service_context, storage_context=storage_context)

custom_query_engines = {
    index.index_id: index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
    )
}

graph2 = ComposableGraph.from_indices(
    GPTTreeIndex,
    [index, empty_index],
    index_summaries=[index_summary, empty_index_summary]
)

query_engine = graph2.as_query_engine(
    custom_query_engines=custom_query_engines
)

response = query_engine.query(
    "What did the author do growing up?", 
)

print(str(response))


response2 = query_engine.query(
    "Who is Obama?", 
)
print(str(response2))