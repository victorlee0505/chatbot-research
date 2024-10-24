# import logging
# import os
# import sys
# import time

# import numpy as np
# import torch
# from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from llama_index.prompts.prompts import SimpleInputPrompt
# from llama_index.llm_predictor import HuggingFaceLLMPredictor

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler = logging.StreamHandler(stream=sys.stdout)
# handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logging.getLogger().addHandler(handler)

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# # checkpoint 
# checkpoint = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

# doc_path ='./data/doc'

# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """ 

# # This will wrap the default prompts that are internal to llama-index
# query_wrapper_prompt = SimpleInputPrompt("<human>:{query_str}<bot>:")

# # A ChatBot class
# # Build a ChatBot class with all necessary modules to make a complete conversation
# class HuggingfaceChatBot():
#     # initialize
#     def __init__(self, 
#                 checkpoint: str = None,
#                 doc_path: str = None
#     ):
#         self.checkpoint = checkpoint
#         self.chat_history = None
#         self.inputs = None
#         self.user_input_ids = None
#         self.bot_history_ids = None
#         self.end_chat = False
#         self.doc_path = doc_path
#         self.documents = None
#         self.llm_predictor = None
#         self.service_context = None
#         self.index = None
#         self.query_engine = None

#         # greet while starting
#         self.welcome()

#     def welcome(self):
#         logger.info("Initializing ChatBot ...")
#         self.load_model()
#         # some time to get user ready
#         time.sleep(2)
#         logger.info('Type "bye" or "quit" or "exit" to end chat \n')
#         # give time to read what has been printed
#         time.sleep(3)
#         # Greet and introduce
#         greeting = np.random.choice([
#             "Welcome, I am ChatBot, here for your kind service",
#             "Hey, Great day! I am your virtual assistant",
#             "Hello, it's my pleasure meeting you",
#             "Hi, I am a ChatBot. Let's chat!"
#         ])
#         print("<bot>: " + greeting)

#     def load_model(self):
#         logger.info("loading data ...")
#         self.documents = SimpleDirectoryReader(self.doc_path).load_data()
#         logger.info("loaded data ...")

#         logger.info("loading model ...")
#         self.llm_predictor = HuggingFaceLLMPredictor(
#             max_input_size=4096, 
#             max_new_tokens=256,
#             temperature=0.7,
#             do_sample=False,
#             system_prompt=system_prompt,
#             query_wrapper_prompt=query_wrapper_prompt,
#             tokenizer_name=self.checkpoint,
#             model_name=self.checkpoint,
#             device_map="auto",
#             stopping_ids=[50278, 50279, 50277, 1, 0],
#             tokenizer_kwargs={"max_length": 4096},
#             model_kwargs={"offload_folder": "offload", "torch_dtype": torch.bfloat16}
#             # uncomment this if using CUDA to reduce memory usage
#             # model_kwargs={"offload_folder": "offload", "torch_dtype": torch.float16}
#         )
#         logger.info("loaded model ...")

#         logger.info("loading ServiceContext ...")
#         self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, chunk_size_limit=512)
#         logger.info("loaded ServiceContext ...")

#         logger.info("loading VectorStoreIndex ...")
#         self.index = GPTVectorStoreIndex.from_documents(self.documents, service_context=self.service_context)
#         logger.info("loaded VectorStoreIndex ...")

#         logger.info("creating query_engine ...")
#         self.query_engine = self.index.as_query_engine(
#             service_context=self.service_context,
#             similarity_top_k=3,
#             streaming=True,
#         )
#         logger.info("created query_engine ...")

#     def promptWrapper(self, text: str):
#         return "<human>: " + text +"\n<bot>: "

#     def user_input(self):
#         # receive input from user
#         text = input("<human>: ")
#         logger.debug(text)
#         # end conversation if user wishes so
#         if text.lower().strip() in ['bye', 'quit', 'exit']:
#             # turn flag on 
#             self.end_chat=True
#             # a closing comment
#             logger.info('<bot>: See you soon! Bye!')
#             time.sleep(1)
#             logger.info('\nQuitting ChatBot ...')
#         else:
#             # continue chat, preprocess input text
#             # encode the new user input, add the eos_token and return a tensor in Pytorch
#             prompt = self.promptWrapper(text)
#             if self.chat_history is not None:
#                 self.chat_history += prompt
#             else:
#                 # if first entry, initialize bot_input_ids
#                 self.chat_history = prompt

#         self.inputs = text

#     def bot_response(self):
        
#         # define the new chat_history_ids based on the preceding chats
#         # generated a response while limiting the total chat history to 1000 tokens, 
#         bot_output = self.query_engine.query(
#             self.inputs, 
#         )

#         # last ouput tokens from bot
#         # response = tokenizer.decode(output[:, self.new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
#         # outputs = self.tokenizer.decode(bot_output[:, self.chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
#         # in case, bot fails to answer
#         if bot_output == "":
#             bot_output = self.random_response()
#         # print bot response
#         self.chat_history += f"{bot_output}\n"
#         # logger.debug(self.chat_history)
#         print(str(bot_output))
        
#     # in case there is no response from model
#     def random_response(self):
#         return "I don't know", "I am not sure"


# # build a ChatBot object
# bot = HuggingfaceChatBot(checkpoint=checkpoint, doc_path=doc_path)
# # start chatting
# while True:
#     # receive user input
#     bot.user_input()
#     # check whether to end chat
#     if bot.end_chat:
#         break
#     # output bot response
#     bot.bot_response()    
	

# # Happy Chatting! 