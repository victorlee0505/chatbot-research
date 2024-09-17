# from dotenv import load_dotenv
# import logging
# import os
# import sys
# import time

# import numpy as np
# import openai
# from langchain.callbacks import get_openai_callback
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain_community.vectorstores.chroma import Chroma

# from chatbot_research.ingestion.ingest_constants import CHROMA_SETTINGS_AZURE, PERSIST_DIRECTORY_AZURE
# from chatbot_research.ingestion.ingest import Ingestion

# load_dotenv()

# persist_directory = PERSIST_DIRECTORY_AZURE
# target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 2))

# openai.api_key = os.getenv("OPENAI_API_KEY")
# model_name = "gpt-3.5-turbo"  # This will correspond to the custom name you chose for your deployment when you deployed a model.


# # A ChatBot class
# # Build a ChatBot class with all necessary modules to make a complete conversation
# class OpenAiChatBot:
#     # initialize
#     def __init__(
#         self,
#         model: str = None,
#         source_path: str = None,
#         load_data: bool = False,
#         show_stream: bool = False,
#         show_source: bool = False,
#         gui_mode: bool = False,
#         log_to_file: bool = False,
#     ):
#         """
#         Initialize AzureOpenAiChatBot object

#         Parameters:
#         - source_path: optional
#         - open_chat: set True to allow answer outside of the context
#         - load_data: set True if you want to load new / additional data. default skipping ingest data.
#         - show_stream: show_stream
#         - show_source: set True will show source of the completion
#         """
#         self.model = model
#         self.source_path = source_path
#         self.load_data = load_data
#         self.show_stream = show_stream
#         self.show_source = show_source
#         self.gui_mode = gui_mode
#         self.llm = None
#         self.embedding_llm = None
#         self.qa = None
#         self.index = None
#         self.chat_history = []
#         self.inputs = None
#         self.end_chat = False
#         self.log_to_file = log_to_file

#         self.logger = logging.getLogger("chatbot-chroma")
#         self.logger.setLevel(logging.INFO)
#         self.logger.propagate = False
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.INFO)
#         formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#         ch.setFormatter(formatter)
#         self.logger.addHandler(ch)

#         if self.log_to_file:
#             log_dir = "logs"
#             os.makedirs(log_dir, exist_ok=True)  
#             log_filename = f"{log_dir}/{self.llm_config.model}.log"
#             fh = logging.FileHandler(log_filename)
#             fh.setLevel(logging.INFO)
#             self.logger.addHandler(fh)
#         # greet while starting
#         if self.model is None or len(self.model) == 0:
#             self.model = model_name
#         self.welcome()

#     def welcome(self):
#         self.logger.info("Initializing ChatBot ...")
#         self.ingest_documents()
#         self.initialize_model()

#         # some time to get user ready
#         time.sleep(2)
#         self.logger.info('Type "bye" or "quit" or "exit" to end chat \n')
#         # give time to read what has been printed
#         time.sleep(3)
#         # Greet and introduce
#         greeting = np.random.choice(
#             [
#                 "Welcome, I am ChatBot, here for your kind service",
#                 "Hey, Great day! I am your virtual assistant",
#                 "Hello, it's my pleasure meeting you",
#                 "Hi, I am a ChatBot. Let's chat!",
#             ]
#         )
#         print("<bot>: " + greeting)

#     def ingest_documents(self):
#         if self.load_data:
#             Ingestion(openai=True, source_path=self.source_path)
#         else:
#             if os.path.exists(persist_directory):
#                 if os.listdir(persist_directory):
#                     self.logger.info(f"Ingestion skipped!")
#                 else:
#                     self.logger.info("PERSIST_DIRECTORY is empty.")
#                     Ingestion(source_path=self.source_path)
#             else:
#                 self.logger.info("PERSIST_DIRECTORY does not exist.")
#                 Ingestion(source_path=self.source_path)

#     def initialize_model(self):
#         self.logger.info("Initializing Model ...")
#         self.embedding_llm = OpenAIEmbeddings(
#             model="text-embedding-ada-002",
#             openai_api_key=openai.api_key,
#         )
#         db = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=self.embedding_llm,
#             client_settings=CHROMA_SETTINGS_AZURE,
#         )
#         retriever = db.as_retriever(
#             search_type="similarity", search_kwargs={"k": target_source_chunks}, max_tokens_limit=1000
#         )
#         # activate/deactivate the streaming StdOut callback for LLMs
#         callbacks = [StreamingStdOutCallbackHandler()] if self.show_stream else []

#         self.llm = ChatOpenAI(
#             model_name=self.model,
#             callbacks=callbacks,
#             openai_api_key=openai.api_key,
#             max_tokens=2000,
#         )

#         memory = ConversationSummaryBufferMemory(
#             llm=self.llm,
#             max_token_limit=1000,
#             output_key="answer",
#             memory_key="chat_history",
#             ai_prefix="<bot>: ",
#             human_prefix="<human>: ",
#         )

#         self.qa = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=retriever,
#             memory=memory,
#             get_chat_history=lambda h: h,
#             return_source_documents=self.show_source,
#             verbose=False,
#         )

#     def count_tokens(self, chain, query):
#         with get_openai_callback() as cb:
#             result = chain.run(query)
#             print(f"Spent a total of {cb.total_tokens} tokens")
#         return result
    
#     def promptWrapper(self, text: str):
#         return "<human>: " + text + "\n<bot>: "

#     def user_input(self, prompt: str = None):
#         # receive input from user
#         if prompt:
#             text = prompt
#         else:
#             text = input("<human>: ")

#         self.logger.debug(text)
#         # end conversation if user wishes so
#         if text.lower().strip() in ["bye", "quit", "exit"] and not self.gui_mode:
#             # turn flag on
#             self.end_chat = True
#             # a closing comment
#             self.logger.info("<bot>: See you soon! Bye!")
#             time.sleep(1)
#             self.logger.info("\nQuitting ChatBot ...")
#             self.inputs = text
#         else:
#             self.inputs = text

#     def bot_response(self) -> str:
#         if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.gui_mode:
#             # a closing comment
#             answer = "<bot>: See you soon! Bye!"
#             print(f"<bot>: {answer}")
#             return answer
#         response = self.qa({"question": self.inputs})
#         answer, docs = (
#             response["answer"],
#             response["source_documents"] if self.show_source else [],
#         )
#         # in case, bot fails to answer
#         if answer == "":
#             answer = self.random_response()
#         # print bot response
#         self.chat_history.append((self.inputs, answer))
#         # logger.debug(self.chat_history)
#         print(f"<bot>: {answer}")
#         if self.show_source:
#             for document in docs:
#                 print(f"<bot>: source_documents")
#                 print("\n> " + document.metadata["source"] + ":")
#                 print(document.page_content)
#         return answer

#     # in case there is no response from model
#     def random_response(self):
#         return "I don't know", "I am not sure"


# if __name__ == "__main__":
#     # build a ChatBot object

#     # Context only chat
#     # bot = AzureOpenAiChatBot()

#     # Open chat
#     bot = OpenAiChatBot()

#     # start chatting
#     while True:
#         # receive user input
#         bot.user_input()
#         # check whether to end chat
#         if bot.end_chat:
#             break
#         # output bot response
#         bot.bot_response()
#     # Happy Chatting!
