import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import torch
# from llama_index import GPTVectorStoreIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader
# from llama_index.llm_predictor import HuggingFaceLLMPredictor
# from llama_index.prompts.prompts import SimpleInputPrompt
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


stop_words = ["<human>:"]


# comment this to use CUDA
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.bfloat16)
# infer
prompt = "<human>: Do you know java programming language?\n<bot>:"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_length = inputs.input_ids.shape[1]
# outputs = model.generate(
#     **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
# )
stop_words_ids = [
    tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
    for stop_word in stop_words
]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
outputs = model.generate(
    **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True, stopping_criteria=StoppingCriteriaList([stopping_criteria])
)
token = outputs.sequences[0, input_length:]
output_str = tokenizer.decode(token)
print(output_str)

prompt = "<human>: Can you write a for loop then?\n<bot>:"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_length = inputs.input_ids.shape[1]
outputs = model.generate(
    **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True, stopping_criteria=StoppingCriteriaList([stopping_criteria])
)
token = outputs.sequences[0, input_length:]
output_str = tokenizer.decode(token)
print(output_str)