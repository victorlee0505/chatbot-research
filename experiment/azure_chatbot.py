import logging
import os
import sys
import time

import numpy as np
import openai
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding, SimpleDirectoryReader
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

openai.api_type = "azure"
openai.api_base = "https://cog-frutomt5wmbzu.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY")
deployment_name = "chat"

# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 50
# set maximum chunk overlap
max_chunk_overlap = 20

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class AzureOpenAiChatBot():
    # initialize
    def __init__(self,
                documents
    ):
        self.llm = AzureOpenAI(deployment_name=deployment_name, model_kwargs={
            "api_key": openai.api_key,
            "api_base": openai.api_base,
            "api_type": openai.api_type,
            "api_version": openai.api_version,
        })
        self.llm_predictor = LLMPredictor(llm=self.llm)
        self.embedding_llm = LangchainEmbedding(
            OpenAIEmbeddings(
                model="text-embedding-ada-002",
                deployment="text-embedding-ada-002",
                openai_api_key= openai.api_key,
                openai_api_base=openai.api_base,
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
            ),
            embed_batch_size=1,
        )
        self.prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=self.llm_predictor,
            embed_model=self.embedding_llm,
            prompt_helper=self.prompt_helper
        )
        self.index = GPTVectorStoreIndex.from_documents(documents, service_context=self.service_context)
        self.query_engine = self.index.as_query_engine()
        self.chat_history = None
        self.inputs = None
        self.end_chat = False
        # greet while starting
        self.welcome()
        
    def welcome(self):
        logger.info("Initializing ChatBot ...")
        # some time to get user ready
        time.sleep(2)
        logger.info('Type "bye" or "quit" or "exit" to end chat \n')
        # give time to read what has been printed
        time.sleep(3)
        # Greet and introduce
        greeting = np.random.choice([
            "Welcome, I am ChatBot, here for your kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a ChatBot. Let's chat!"
        ])
        print("<bot>: " + greeting)
        
    def promptWrapper(self, text: str):
        return "<human>: " + text +"\n<bot>: "
    
    def user_input(self):
        # receive input from user
        text = input("<human>: ")
        logger.debug(text)
        # end conversation if user wishes so
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            # turn flag on 
            self.end_chat=True
            # a closing comment
            logger.info('<bot>: See you soon! Bye!')
            time.sleep(1)
            logger.info('\nQuitting ChatBot ...')
        else:
            prompt = self.promptWrapper(text)
            if self.chat_history is not None:
                self.chat_history += prompt
            else:
                self.chat_history = prompt

            self.inputs = text
            

    def bot_response(self):
        
        response = self.query_engine.query(self.inputs)

        # in case, bot fails to answer
        if response == "":
            response = self.random_response()
        # print bot response
        self.chat_history += f"{response}\n"
        # logger.debug(self.chat_history)
        print(f"<bot>: {response}")
        
    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


documents = SimpleDirectoryReader("./data/excel").load_data()
# build a ChatBot object
bot = AzureOpenAiChatBot(documents = documents)
# start chatting
while True:
    # receive user input
    bot.user_input()
    # check whether to end chat
    if bot.end_chat:
        break
    # output bot response
    bot.bot_response()    
	

# Happy Chatting! 