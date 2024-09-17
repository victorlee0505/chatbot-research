# import logging
# import os
# import sys
# import time
# from typing import Dict, Union, Any, List

# from ctransformers import (
#     AutoModelForCausalLM as cAutoModelForCausalLM,
#     AutoTokenizer as cAutoTokenizer,
# )

# import numpy as np
# import torch
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.chains import ConversationChain, LLMChain
# from langchain.memory import ConversationSummaryBufferMemory
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     GenerationConfig,
#     StoppingCriteria,
#     StoppingCriteriaList,
#     TextStreamer,
#     TextIteratorStreamer,
#     pipeline,
# )
# from chatbot_research.huggingface.config.hf_llm_config import LLMConfig
# from chatbot_research.huggingface.config.hf_prompts import NO_MEM_PROMPT

# class MyCustomHandler(BaseCallbackHandler):
#     def on_llm_start(
#         self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
#     ) -> Any:
#         """Run when LLM starts running."""
#         print(f"My custom handler, llm_start: {prompts[-1]} stop")

#     def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
#         """Run when LLM ends running."""
#         print(f"My custom handler, llm_end: {response.generations[0][0].text} stop")

# class StoppingCriteriaSub(StoppingCriteria):
#     def __init__(self, stops=[], encounters=1):
#         super().__init__()
#         self.stops = stops

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         for stop in self.stops:
#             if torch.all((stop == input_ids[0][-len(stop) :])).item():
#                 return True
#         return False
    
# def timer_decorator(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()  # start time before function executes
#         result = func(*args, **kwargs)  # execute function
#         end_time = time.time()  # end time after function executes
#         exec_time = end_time - start_time  # execution time
#         args[0].logger.info(f"Executed {func.__name__} in {exec_time:.4f} seconds")
#         return result
#     return wrapper

# # A ChatBot class
# # Build a ChatBot class with all necessary modules to make a complete conversation
# class HuggingFaceLLM:
#     # initialize
#     def __init__(
#         self,
#         llm_config: LLMConfig = None,
#         show_callback: bool = False,
#         disable_mem: bool = False,
#         gpu_layers: int = 0,
#         gpu: bool = False,
#         server_mode: bool = False,
#         log_to_file: bool = False,
#     ):
#         self.llm_config = llm_config
#         self.show_callback = show_callback
#         self.disable_mem = disable_mem
#         self.gpu_layers = gpu_layers
#         self.gpu = gpu
#         self.device = None
#         self.server_mode = server_mode
#         self.llm = None
#         self.tokenizer = None
#         self.model = None
#         self.pipeline = None
#         self.streamer = None
#         self.qa = None
#         self.chat_history = []
#         self.inputs = None
#         self.end_chat = False
#         self.log_to_file = log_to_file

#         self.logger = logging.getLogger("chatbot-base")
#         self.logger.setLevel(logging.INFO)
#         self.logger.propagate = False
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.INFO)
#         formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#         ch.setFormatter(formatter)
#         self.logger.addHandler(ch)

#         if self.log_to_file:
#             log_dir = "logs"
#             try:
#                 os.makedirs(log_dir, exist_ok=True)
#             except Exception as e:
#                 print(f"Error creating directory: {e}")
#             filename = self.llm_config.model.replace("/", "_")
#             print(f"Logging to file: {filename}.log")
#             log_filename = f"{log_dir}/{filename}.log"
#             fh = logging.FileHandler(log_filename)
#             fh.setLevel(logging.INFO)
#             self.logger.addHandler(fh)

#         # greet while starting
#         self.welcome()

#     def welcome(self) -> HuggingFacePipeline:
#         if self.llm_config:
#             self.llm_config.validate()
#         self.logger.info("Initializing ChatBot ...")
#         if self.llm_config.model_type is None:
#             torch.set_num_threads(os.cpu_count())
#             if self.gpu:
#                 self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
#             else:
#                 self.logger.info("Disable CUDA")
#                 os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#                 self.device=torch.device('cpu')
#             return self.initialize_model()
#         else:
#             return self.initialize_gguf_model()

#     def initialize_model(self) -> HuggingFacePipeline:
#         self.logger.info("Initializing Model ...")
#         try:
#             generation_config = GenerationConfig.from_pretrained(self.llm_config.model)
#         except Exception as e:
#             generation_config = None
#         self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model, model_max_length=self.llm_config.model_max_length)
#         if self.server_mode:
#             self.streamer = TextIteratorStreamer(self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
#         else:
#             self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
#         if self.gpu:
#             self.model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
#             self.model.half().cuda()
#             torch_dtype = torch.float16
#         else:
#             self.model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
#             torch_dtype = torch.bfloat16

#         if self.gpu:
#             stop_words_ids = [
#                 self.tokenizer(stop_word, return_tensors="pt").to('cuda')["input_ids"].squeeze()
#                 for stop_word in self.llm_config.stop_words
#             ]
#         else:
#             stop_words_ids = [
#                 self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
#                 for stop_word in self.llm_config.stop_words
#             ]
#         stopping_criteria = StoppingCriteriaList(
#             [StoppingCriteriaSub(stops=stop_words_ids)]
#         )

#         pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=self.llm_config.max_new_tokens,
#             temperature=self.llm_config.temperature,
#             top_p=self.llm_config.top_p,
#             top_k=self.llm_config.top_k,
#             generation_config=generation_config,
#             repetition_penalty=1.2,
#             pad_token_id=self.tokenizer.eos_token_id,
#             device=self.device,
#             do_sample=self.llm_config.do_sample,
#             torch_dtype=torch_dtype,
#             stopping_criteria=stopping_criteria,
#             streamer=self.streamer,
#             model_kwargs={"offload_folder": "offload"},
#         )
#         handler = []
#         handler = handler.append(MyCustomHandler()) if self.show_callback else handler
#         self.pipeline = HuggingFacePipeline(pipeline=pipe, callbacks=handler)
#         return self.pipeline

#     def initialize_gguf_model(self) -> HuggingFacePipeline:
#         self.logger.info("Initializing Model ...")

#         model = cAutoModelForCausalLM.from_pretrained(self.llm_config.model, 
#             model_file=self.llm_config.model_file, 
#             model_type=self.llm_config.model_type,
#             hf=True,
#             temperature=self.llm_config.temperature,
#             top_p=self.llm_config.top_p,
#             top_k=self.llm_config.top_k,
#             repetition_penalty=1.2,
#             context_length=self.llm_config.model_max_length,
#             max_new_tokens=self.llm_config.max_new_tokens,
#             # stop=self.llm_config.stop_words,
#             threads=os.cpu_count(),
#             stream=True,
#             gpu_layers=self.gpu_layers
#             )
#         self.tokenizer = cAutoTokenizer.from_pretrained(model)

#         if self.server_mode:
#             self.streamer = TextIteratorStreamer(self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
#         else:
#             self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)

#         stop_words_ids = [
#                 self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
#                 for stop_word in self.llm_config.stop_words
#             ]
#         stopping_criteria = StoppingCriteriaList(
#             [StoppingCriteriaSub(stops=stop_words_ids)]
#         )
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=self.llm_config.max_new_tokens,
#             pad_token_id=self.tokenizer.eos_token_id,
#             # eos_token_id=self.tokenizer.convert_tokens_to_ids(self.llm_config.eos_token_id) if self.llm_config.eos_token_id is not None else None,
#             stopping_criteria=stopping_criteria,
#             streamer=self.streamer,
#             model_kwargs={"offload_folder": "offload"},
#         )
#         handler = []
#         handler = handler.append(MyCustomHandler()) if self.show_callback else handler
#         self.pipeline = HuggingFacePipeline(pipeline=pipe, callbacks=handler)
#         return self.pipeline