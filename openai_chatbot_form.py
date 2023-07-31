import logging
import json
import os
import sys
import time

import numpy as np
import openai
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from PersonalDetails import PersonalDetails

from constants import CHROMA_SETTINGS_AZURE, PERSIST_DIRECTORY_AZURE
from ingest import Ingestion

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
# handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

persist_directory = PERSIST_DIRECTORY_AZURE
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 2))

openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"  # This will correspond to the custom name you chose for your deployment when you deployed a model.


# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class OpenAiChatBot:
    # initialize
    def __init__(
        self,
        model: str = None,
        source_path: str = None,
        load_data: bool = False,
        show_stream: bool = False,
        show_source: bool = False,
        gui_mode: bool = False,
        user_details: PersonalDetails = None
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
        self.model = model
        self.source_path = source_path
        self.load_data = load_data
        self.show_stream = show_stream
        self.show_source = show_source
        self.gui_mode = gui_mode
        self.llm = None
        self.embedding_llm = None
        self.qa = None
        self.index = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        self.user_details = user_details
        # greet while starting
        if self.model is None or len(self.model) == 0:
            self.model = model_name

        if self.user_details is None:
            self.user_details = PersonalDetails(first_name="",
                                last_name="",
                                full_name="",
                                city="",
                                email="",
                                language="")
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

        self.llm = ChatOpenAI(
            model_name=self.model,
            temperature=0,
            callbacks=callbacks,
            openai_api_key=openai.api_key,
            max_tokens=2000,
        )

        PROMPT = ChatPromptTemplate.from_template(
            "Below is are some things to ask the user for in a coversation way. you should only ask one question at a time even if you don't get all the info \
            don't ask as a list! Don't greet the user! Don't say Hi.Explain you need to get some info. If the ask_for list is empty then thank them and ask how you can help them \n\n \
            ### ask_for list: {ask_for}"
        )

        self.qa = LLMChain(llm=self.llm, prompt=PROMPT, verbose=False)

    def count_tokens(self, chain, query):
        with get_openai_callback() as cb:
            result = chain.run(query)
            print(f"Spent a total of {cb.total_tokens} tokens")
        return result
    
    def promptWrapper(self, text: str):
        return "<human>: " + text + "\n<bot>: "
    
    def check_what_is_empty(self, user_peronal_details):
        ask_for = []
        # Check if fields are empty
        for field, value in user_peronal_details.dict().items():
            if value in [None, "", 0]:  # You can add other 'empty' conditions as per your requirements
                print(f"Field '{field}' is empty.")
                ask_for.append(f'{field}')
        return ask_for

    def add_non_empty_details(self, current_details: PersonalDetails, new_details: PersonalDetails):
        non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
        updated_details = current_details.copy(update=non_empty_details)
        return updated_details
    
    def filter_response(self, text_input, user_details):
        chain = create_tagging_chain_pydantic(pydantic_schema=PersonalDetails, llm=self.llm)
        res = chain.run(text_input)
        # add filtered info to the
        user_details = self.add_non_empty_details(user_details,res)
        ask_for = self.check_what_is_empty(user_details)
        return user_details, ask_for
    
    def ask_for_info(self, ask_for = ['name']):
        ai_chat = self.qa.run(ask_for=ask_for)
        return ai_chat
    
    def generate_form(self):
        dir_path = 'tmp'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Specify the file path
        file_path = os.path.join(dir_path, 'user_details_Form.json')
        with open(file_path, 'w') as f:
                json.dump(self.user_details.dict(), f)

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
        new_user_details, ask_for = self.filter_response(text_input=self.inputs, user_details=self.user_details)
        self.user_details = new_user_details
        if ask_for:
            answer = self.ask_for_info(ask_for)
        else:
            print('Everything gathered, generating form.\n')
            self.generate_form()
            self.end_chat = True
            answer = "Form generated, goodbye!"
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
    bot = OpenAiChatBot()
    first_prompt = bot.ask_for_info()
    print(f"<bot>: {first_prompt}")
    # start chatting
    while True:
        # receive user input
        bot.user_input()
        # check whether to end chat
        if bot.end_chat:
            break
        # output bot response
        bot.bot_response()
        if bot.end_chat:
            break
    # Happy Chatting!
