from dotenv import load_dotenv
import logging
import os
import sys
import time

import numpy as np
import openai
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # this may change in the future
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # This will correspond to the custom name you chose for your deployment when you deployed a model.

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class AzureOpenAiChatBotBase:
    # initialize
    def __init__(
        self,
        show_stream: bool = False,
        gui_mode: bool = False,
        log_to_file: bool = False,
    ):
        self.llm = None
        self.show_stream = show_stream
        self.gui_mode = gui_mode
        self.qa = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        self.log_to_file = log_to_file

        self.logger = logging.getLogger("chatbot-chroma")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if self.log_to_file:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)  
            log_filename = f"{log_dir}/{self.llm_config.model}.log"
            fh = logging.FileHandler(log_filename)
            fh.setLevel(logging.INFO)
            self.logger.addHandler(fh)
        # greet while starting
        self.welcome()

    def welcome(self):
        self.logger.info("Initializing ChatBot ...")
        self.initialize_model()
        # some time to get user ready
        time.sleep(2)
        self.logger.info('Type "bye" or "quit" or "exit" to end chat \n')
        # give time to read what has been printed
        time.sleep(3)
        # Greet and introduce
        greeting = np.random.choice(
            [
                "Welcome, I am ChatBot, here for your kind service",
                "Hey, Great day! I am your virtual assistant",
                "Hello, it's my pleasure meeting you",
                "Hi, I am a ChatBot. Let's chat!",
            ]
        )
        print("<bot>: " + greeting)

    def initialize_model(self):
        self.logger.info("Initializing Model ...")
        callbacks = []
        callbacks = callbacks.append(StreamingStdOutCallbackHandler()) if self.show_stream else callbacks
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_type=openai.api_type,
            openai_api_base=openai.api_base,
            openai_api_version=openai.api_version,
            openai_api_key=openai.api_key,
            streaming=True,
        )

        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
        )
        self.qa = ConversationChain(llm=self.llm, memory=memory, verbose=False, callbacks=callbacks)

    def count_tokens(self, chain, query):
        with get_openai_callback() as cb:
            result = chain.run(query)
            print(f"Spent a total of {cb.total_tokens} tokens")
        return result

    def user_input(self, prompt: str = None):
        # receive input from user
        if prompt:
            text = prompt
        else:
            text = input("<human>: ")

        self.logger.debug(text)
        # end conversation if user wishes so
        if text.lower().strip() in ["bye", "quit", "exit"] and not self.gui_mode:
            # turn flag on
            self.end_chat = True
            # a closing comment
            self.logger.info("<bot>: See you soon! Bye!")
            time.sleep(1)
            self.logger.info("\nQuitting ChatBot ...")
            self.inputs = text
        else:
            self.inputs = text

    def bot_response(self) -> str:
        if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.gui_mode:
            # a closing comment
            answer = "<bot>: See you soon! Bye!"
            print(f"<bot>: {answer}")
            return answer
        answer = self.qa.run(self.inputs)
        # in case, bot fails to answer
        if answer == "":
            answer = self.random_response()
        # print bot response
        self.chat_history.append((self.inputs, answer))
        # logger.debug(self.chat_history)
        print(f"<bot>: {answer}")
        self.count_tokens(self.qa, self.inputs)
        return answer

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":
    # build a ChatBot object
    bot = AzureOpenAiChatBotBase()
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
