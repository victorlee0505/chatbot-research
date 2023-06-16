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

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
# handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

openai.api_type = "azure"
openai.api_base = os.getenv(
    "OPENAI_AZURE_BASE_URL"
)  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_version = "2023-05-15"  # this may change in the future
openai.api_key = os.getenv("OPENAI_API_KEY")
deployment_name = os.getenv(
    "OPENAI_DEPLOYMENT_NAME"
)  # This will correspond to the custom name you chose for your deployment when you deployed a model.


# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class AzureOpenAiChatBotBase:
    # initialize
    def __init__(
        self,
        show_stream: bool = False,
        gui_mode: bool = False,
    ):
        self.llm = None
        self.show_stream = show_stream
        self.gui_mode = gui_mode
        self.qa = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        # greet while starting
        self.welcome()

    def welcome(self):
        logger.info("Initializing ChatBot ...")
        self.initialize_model()
        # some time to get user ready
        time.sleep(2)
        logger.info('Type "bye" or "quit" or "exit" to end chat \n')
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
        logger.info("Initializing Model ...")
        callbacks = [StreamingStdOutCallbackHandler()] if self.show_stream else []
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            callbacks=callbacks,
            openai_api_type=openai.api_type,
            openai_api_base=openai.api_base,
            openai_api_version=openai.api_version,
            openai_api_key=openai.api_key,
        )

        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
        )
        self.qa = ConversationChain(llm=self.llm, memory=memory, verbose=False)

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

        logger.debug(text)
        # end conversation if user wishes so
        if text.lower().strip() in ["bye", "quit", "exit"] and not self.gui_mode:
            # turn flag on
            self.end_chat = True
            # a closing comment
            logger.info("<bot>: See you soon! Bye!")
            time.sleep(1)
            logger.info("\nQuitting ChatBot ...")
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
