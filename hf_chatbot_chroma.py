import logging
import os
import sys
import time
from typing import Dict, Union, Any, List
import chromadb

import numpy as np
import torch
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains.chat_vector_db.prompts import QA_PROMPT
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, pipeline)
from langchain.prompts.prompt import PromptTemplate

from ingest_constants import CHROMA_SETTINGS_HF, PERSIST_DIRECTORY_HF
from hf_llm_config import LLMConfig
from ingest import Ingestion

from hf_llm_config import REDPAJAMA_3B, REDPAJAMA_7B, VICUNA_7B, LLMConfig

persist_directory = PERSIST_DIRECTORY_HF

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print(f"My custom handler, llm_start: {prompts[-1]} stop")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print(f"My custom handler, llm_end: {response.generations[0][0].text} stop")

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # start time before function executes
        result = func(*args, **kwargs)  # execute function
        end_time = time.time()  # end time after function executes
        exec_time = end_time - start_time  # execution time
        args[0].logger.info(f"Executed {func.__name__} in {exec_time:.4f} seconds")
        return result
    return wrapper

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class HuggingFaceChatBotChroma:
    # initialize
    def __init__(
        self,
        llm_config: LLMConfig = None,
        load_data: bool = False,
        show_stream: bool = False,
        show_source: bool = False,
        show_callback: bool = False,
        gpu: bool = False,
        gui_mode: bool = False,
        log_to_file: bool = False,
    ):
        self.llm_config = llm_config
        self.load_data = load_data
        self.show_stream = show_stream
        self.show_source = show_source
        self.show_callback = show_callback
        self.gpu = gpu
        self.device = None
        self.gui_mode = gui_mode
        self.llm = None
        self.embedding_llm = None
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
        if self.llm_config:
            self.llm_config.validate()
        self.logger.info("Initializing ChatBot ...")
        torch.set_num_threads(os.cpu_count())
        if not self.gpu:
            torch.device('cpu')
            self.logger.info("Disable CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device=torch.device('cpu')
        else:
            torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.ingest_documents()
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

    def ingest_documents(self):
        offline = True
        if self.load_data:
            Ingestion(offline=offline)
        else:
            if os.path.exists(persist_directory):
                if os.listdir(persist_directory):
                    self.logger.info(f"Ingestion skipped!")
                else:
                    self.logger.info("PERSIST_DIRECTORY is empty.")
                    Ingestion(offline=offline)
            else:
                self.logger.info("PERSIST_DIRECTORY does not exist.")
                Ingestion(offline=offline)

    def initialize_model(self):
        self.logger.info("Initializing Model ...")
        if self.gpu:
            self.embedding_llm = HuggingFaceEmbeddings()
        else:
            model_kwargs = {'device': 'cpu'}
            self.embedding_llm = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
        client = chromadb.PersistentClient(settings=CHROMA_SETTINGS_HF, path=persist_directory)
        db = Chroma(
            client=client,
            embedding_function=self.embedding_llm,
        )
        retriever = db.as_retriever(
            search_type="similarity", search_kwargs={"k": self.llm_config.target_source_chunks}, max_tokens_limit=self.llm_config.retriever_max_tokens_limit
        )

        tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model, model_max_length=self.llm_config.model_max_length)
        if self.gpu:
            model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
            model.half().cuda()
            torch_dtype = torch.float16
        else:
            model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
            torch_dtype = torch.bfloat16

        if self.gpu:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt").to("cuda")["input_ids"].squeeze()
                for stop_word in self.llm_config.stop_words
            ]
        else:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
                for stop_word in self.llm_config.stop_words
            ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.llm_config.max_new_tokens,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p,
            top_k=self.llm_config.top_k,
            pad_token_id=tokenizer.eos_token_id,
            device=self.device,
            do_sample=self.llm_config.do_sample,
            torch_dtype=torch_dtype,
            stopping_criteria=stopping_criteria,
            model_kwargs={"offload_folder": "offload"},
        )

        handler = [MyCustomHandler()] if self.show_callback else None
        self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=500,
            output_key="answer",
            memory_key="chat_history",
            ai_prefix=self.llm_config.ai_prefix,
            human_prefix=self.llm_config.human_prefix,
        )

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            memory=memory,
            # combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            get_chat_history=lambda h: h,
            return_source_documents=self.show_source,
        )

    def user_input(self, prompt: str = None):
        # receive input from user
        if prompt:
            text = prompt
        else:
            text = input("<human>: ")

        # end conversation if user wishes so
        if text.lower().strip() in ["bye", "quit", "exit"] and not self.gui_mode:
            # turn flag on
            self.end_chat = True
            # a closing comment
            print("<bot>: See you soon! Bye!")
            time.sleep(1)
            self.logger.info("\nQuitting ChatBot ...")
            self.inputs = text
        elif text.lower().strip() in ["reset"]:
            self.logger.info("<bot>: reset conversation memory detected.")
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.llm_config.max_new_tokens,
                output_key="answer",
                memory_key="chat_history",
                ai_prefix=self.llm_config.ai_prefix,
                human_prefix=self.llm_config.human_prefix,
            )
            self.qa.memory = memory
            self.inputs = text
        else:
            self.inputs = text

    @timer_decorator
    def bot_response(self) -> str:
        if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.gui_mode:
            # a closing comment
            answer = "<bot>: See you soon! Bye!"
            print(f"<bot>: {answer}")
            return answer
        if self.inputs.lower().strip() in ["reset"]:
            # a closing comment
            answer = "<bot>: Conversation Memory cleared!"
            print(f"<bot>: {answer}")
            return answer
        response = self.qa({"question": self.inputs})
        answer, docs = (
            response["answer"],
            response["source_documents"] if self.show_source else [],
        )
        # in case, bot fails to answer
        if answer == "":
            answer = self.random_response()
        else:
            answer = answer.replace("\n<human>:", "")
        # print bot response
        self.chat_history.append((f"<human>: {self.inputs}", f"<bot>: {answer}"))
        # logger.info(self.chat_history)
        print(f"<bot>: {answer}")
        if self.show_source:
            print(f"<bot>: source_documents")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
        return answer

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":
    # build a ChatBot object
    bot = HuggingFaceChatBotChroma(llm_config=REDPAJAMA_3B, show_source=True)
    # bot = HuggingFaceChatBotChroma(llm_config=REDPAJAMA_7B, show_source=True)
    # bot = HuggingFaceChatBotChroma(llm_config=VICUNA_7B)
    
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
