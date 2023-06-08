import logging
import os
import sys
import time

import numpy as np
import openai
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores import Chroma

from ingest import Ingestion
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
# handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

persist_directory = PERSIST_DIRECTORY
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_AZURE_BASE_URL")  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_version = "2023-05-15"  # this may change in the future
openai.api_key = os.getenv("OPENAI_API_KEY")
deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")  # This will correspond to the custom name you chose for your deployment when you deployed a model.

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class AzureOpenAiChatBot:
    # initialize
    def __init__(
        self,
        source_path: str = None,
        open_chat: bool = False,
        load_data: bool = False,
        show_stream: bool = False,
        show_source: bool = False,
    ):
        """
        Initialize AzureOpenAiChatBot object

        Parameters:
        - source_path: optional
        - open_chat: set True to allow answer outside of the context
        - load_data: set True if you want to load new / additional data. default skipping ingest data.
        - show_stream: show_stream
        - show_source: set True will show source of the completion
        """
        self.source_path = source_path
        self.open_chat = open_chat
        self.load_data = load_data
        self.show_stream = show_stream
        self.show_source = show_source
        self.llm = None
        self.embedding_llm = None
        self.qa = None
        self.index = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        # greet while starting
        self.welcome()

    def welcome(self):
        logger.info("Initializing ChatBot ...")
        self.ingest_documents()
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

    def ingest_documents(self):
        if self.load_data:
            Ingestion(source_path=self.source_path)
        else:
            if os.path.exists(PERSIST_DIRECTORY):
                if os.listdir(PERSIST_DIRECTORY):
                    logger.info(f"Ingestion skipped!")
                else:
                    logger.info("PERSIST_DIRECTORY is empty.")
                    Ingestion(source_path=self.source_path)
            else:
                logger.info("PERSIST_DIRECTORY does not exist.")
                Ingestion(source_path=self.source_path)

    def initialize_model(self):
        logger.info("Initializing Model ...")
        self.embedding_llm = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            deployment="text-embedding-ada-002",
            openai_api_key=openai.api_key,
            openai_api_base=openai.api_base,
            openai_api_type=openai.api_type,
            openai_api_version=openai.api_version,
        )
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_llm,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        # activate/deactivate the streaming StdOut callback for LLMs
        callbacks = [StreamingStdOutCallbackHandler()] if self.show_stream else []

        if self.open_chat:
            logger.info(f"Open Chat = True!")
            self.llm = AzureChatOpenAI(
                deployment_name=deployment_name,
                callbacks=callbacks,
                openai_api_type=openai.api_type,
                openai_api_base=openai.api_base,
                openai_api_version=openai.api_version,
                openai_api_key=openai.api_key,
            )
        else:
            logger.info(f"Open Chat = False!")
            self.llm = AzureOpenAI(
                deployment_name=deployment_name,
                callbacks=callbacks,
                model_kwargs={
                    "api_key": openai.api_key,
                    "api_base": openai.api_base,
                    "api_type": openai.api_type,
                    "api_version": openai.api_version,
                },
            )

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            return_source_documents=self.show_source,
            verbose=False,
        )

    def promptWrapper(self, text: str):
        return "<human>: " + text + "\n<bot>: "

    def user_input(self):
        # receive input from user
        text = input("<human>: ")
        logger.debug(text)
        # end conversation if user wishes so
        if text.lower().strip() in ["bye", "quit", "exit"]:
            # turn flag on
            self.end_chat = True
            # a closing comment
            logger.info("<bot>: See you soon! Bye!")
            time.sleep(1)
            logger.info("\nQuitting ChatBot ...")
        else:
            self.inputs = text

    def bot_response(self):
        response = self.qa({"question": self.inputs, "chat_history": self.chat_history})
        answer, docs = (
            response["answer"],
            response["source_documents"] if self.show_source else [],
        )
        # in case, bot fails to answer
        if answer == "":
            answer = self.random_response()
        # print bot response
        self.chat_history.append((self.inputs, answer))
        # logger.debug(self.chat_history)
        print(f"<bot>: {answer}")
        if self.show_source:
            for document in docs:
                print(f"<bot>: source_documents")
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":
    # build a ChatBot object

    # Context only chat
    # bot = AzureOpenAiChatBot()

    # Open chat
    bot = AzureOpenAiChatBot(open_chat=True)
    
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
