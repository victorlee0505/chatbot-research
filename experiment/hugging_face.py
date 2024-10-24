# import os
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# import torch
# from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index import LangchainEmbedding
# from llama_index.prompts.prompts import SimpleInputPrompt
# from llama_index.llm_predictor import HuggingFaceLLMPredictor
# from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# # comment this to use CUDA
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# documents = SimpleDirectoryReader('./data/doc').load_data()
# checkpoint = "StabilityAI/stablelm-tuned-alpha-3b"
# checkpoint2 = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
# checkpoint3 = "nomic-ai/gpt4all-j"
# checkpoint3v = "v1.2-jazzy"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # model = AutoModelForCausalLM.from_pretrained(checkpoint)
# embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

# class StoppingCriteriaSub(StoppingCriteria):
#     def __init__(self, stops=[], encounters=1):
#         super().__init__()
#         self.stops = stops

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         for stop in self.stops:
#             if torch.all((stop == input_ids[0][-len(stop) :])).item():
#                 return True

#         return False

# stop_words = ["<human>:"]
# stop_words_ids = [
#     tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
#     for stop_word in stop_words
# ]
# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])



# # This will wrap the default prompts that are internal to llama-index
# query_wrapper_prompt = SimpleInputPrompt("<human>: {query_str}\n<bot>:")

# hf_predictor = HuggingFaceLLMPredictor(
#     max_input_size=4096, 
#     max_new_tokens=256,
#     temperature=0.7,
#     do_sample=False,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
#     model_name="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
#     # tokenizer=tokenizer,
#     # model=model,
#     device_map="auto",
#     # stopping_ids=stopping_criteria,
#     tokenizer_kwargs={"max_length": 4096},
#     model_kwargs={"offload_folder": "offload"}
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"offload_folder": "offload", "torch_dtype": torch.float16}
# )



# # llm_predictor = LLMPredictor(llm=hf_predictor)
# # service_context = ServiceContext.from_defaults(llm_predictor=hf_predictor, chunk_size_limit=512)
# service_context = ServiceContext.from_defaults(llm_predictor=hf_predictor, chunk_size_limit=512, embed_model=embed_model)

# index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# query_engine = index.as_query_engine(
#     retriever_mode="embedding",
#     service_context=service_context,
#     similarity_top_k=3,
#     streaming=True,
# )
# response = query_engine.query(
#     "What did the author do growing up?", 
# )
# response.print_response_stream()
# # print(str(response))