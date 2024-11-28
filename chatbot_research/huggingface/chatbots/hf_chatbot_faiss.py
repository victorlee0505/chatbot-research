import logging
import os
import time
from typing import Any, Dict, List

from langchain_community.vectorstores.faiss import FAISS
import numpy as np
import torch
from langchain.callbacks.base import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import StoppingCriteria

from chatbot_research.huggingface.config.hf_llm_config import LLMConfig
from chatbot_research.huggingface.inference.hf_llama_cpp import HFllamaCpp
from chatbot_research.huggingface.inference.hf_transformer import HFTransformer
from chatbot_research.ingestion.ingest_constants import (
    CHROMA_SETTINGS_HF,
    PERSIST_DIRECTORY_HF,
    STELLA_EN_1_5B_V5,
)
from chatbot_research.ingestion.ingest_faiss import IngestionFAISS

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
class HuggingFaceChatBotFaiss:
    # initialize
    def __init__(
        self,
        llm_config: LLMConfig = None,
        disable_mem: bool = False,
        load_data: bool = False,
        show_stream: bool = False,
        show_source: bool = False,
        show_callback: bool = False,
        gpu_layers: int = 0,
        gpu: bool = False,
        server_mode: bool = False,
        log_to_file: bool = False,
    ):
        self.llm_config = llm_config
        self.disable_mem = disable_mem
        self.load_data = load_data
        self.show_stream = show_stream
        self.show_source = show_source
        self.show_callback = show_callback
        self.gpu_layers = gpu_layers
        self.gpu = gpu
        self.device = None
        self.server_mode = server_mode
        self.llm = None
        self.streamer = None
        self.embedding_llm = None
        self.qa = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        self.log_to_file = log_to_file

        self.logger = logging.getLogger("chatbot-faiss")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if self.log_to_file:
            log_dir = "logs"
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory: {e}")
            filename = self.llm_config.model.replace("/", "_")
            print(f"Logging to file: {filename}.log")
            log_filename = f"{log_dir}/{filename}.log"
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
            torch.device("cpu")
            self.logger.info("Disable CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device = torch.device("cpu")
        else:
            torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
            self.device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        vectorstore: FAISS | None = self.ingest_documents()

        if self.gpu:
            self.embedding_llm = HuggingFaceEmbeddings(model_name=STELLA_EN_1_5B_V5)
        else:
            model_kwargs = {
                "trust_remote_code": True,
                "config_kwargs": {
                    "use_memory_efficient_attention": False,
                    "unpad_inputs": False,
                },
            }
            self.embedding_llm = HuggingFaceEmbeddings(
                model_name=STELLA_EN_1_5B_V5, model_kwargs=model_kwargs
            )

        self.logger.info("Initializing ChatBot ...")
        if self.llm_config.model_type is None:
            self.inference = HFTransformer(
                logger=self.logger,
                llm_config=self.llm_config,
                gpu=self.gpu,
                server_mode=self.server_mode,
                disable_mem=self.disable_mem,
                show_callback=self.show_callback,
            )
            self.qa = self.inference.initialize_retrival_chain(vectorstore)
        else:
            self.inference = HFllamaCpp(
                logger=self.logger,
                llm_config=self.llm_config,
                gpu=self.gpu,
                server_mode=self.server_mode,
                disable_mem=self.disable_mem,
                show_callback=self.show_callback,
            )
            self.qa = self.inference.initialize_retrival_chain(vectorstore)
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

    def ingest_documents(self) -> FAISS | None:
        offline = True
        if self.load_data:
            ingest = IngestionFAISS(offline=offline, embedding_model=STELLA_EN_1_5B_V5)
            return ingest.run_ingest()
        else:
            if os.path.exists(persist_directory):
                if os.listdir(persist_directory):
                    self.logger.info(f"Ingestion skipped!")
                    ingest = IngestionFAISS(offline=offline, embedding_model=STELLA_EN_1_5B_V5)
                    return ingest.load_vectorstore()
                else:
                    self.logger.info("PERSIST_DIRECTORY is empty.")
                    ingest = IngestionFAISS(offline=offline, embedding_model=STELLA_EN_1_5B_V5)
                    return ingest.run_ingest()
            else:
                self.logger.info("PERSIST_DIRECTORY does not exist.")
                ingest = IngestionFAISS(offline=offline, embedding_model=STELLA_EN_1_5B_V5)
                return ingest.run_ingest()

    def user_input(self, prompt: str = None):
        # receive input from user
        if prompt:
            text = prompt
        else:
            text = input("<human>: ")

        # end conversation if user wishes so
        if text.lower().strip() in ["bye", "quit", "exit"] and not self.server_mode:
            # turn flag on
            self.end_chat = True
            # a closing comment
            print("<bot>: See you soon! Bye!")
            time.sleep(1)
            self.logger.info("\nQuitting ChatBot ...")
            self.inputs = text
        elif text.lower().strip() in ["reset"]:
            self.logger.info("<bot>: reset conversation memory detected.")
            self.inputs = text
        else:
            self.inputs = text

    @timer_decorator
    def bot_response(self) -> str:
        if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.server_mode:
            # a closing comment
            answer = "<bot>: See you soon! Bye!"
            print(f"<bot>: {answer}")
            return answer
        if self.inputs.lower().strip() in ["reset"]:
            # a closing comment
            self.qa = self.inference.reset_chat_memory()
            answer = "<bot>: Conversation Memory cleared!"
            print(f"<bot>: {answer}")
            return answer
        response = {}
        answer = ""
        curr_key = None
        previous_length = 0
        # Stream the response
        for chunk in self.qa.stream(
            {
                "input": self.inputs,
                "chat_history": [],
            },
            config={"configurable": {"session_id": "unused"}},
        ):

            for key in chunk:
                if key not in response:
                    response[key] = chunk[key]
                else:
                    response[key] += chunk[key]

                # Check if the key is 'answer' and handle it
                if key == "answer":
                    if previous_length == 0:
                        print(f"<bot>: {chunk[key]}", end="", flush=True)
                    else:
                        print(chunk[key], end="", flush=True)  # Print incrementally
                    previous_length = len(chunk[key])

                # Update curr_key only if it's None or different from the current key
                if curr_key is None or key != curr_key:
                    curr_key = key

        answer = response["answer"]

        # in case, bot fails to answer
        if answer == "":
            answer = self.random_response()
            print(f"<bot>: {answer}")
        # print bot response
        self.chat_history.append((f"<human>: {self.inputs}", f"<bot>: {answer}"))
        # logger.info(self.chat_history)

        if self.show_source:
            print(f"<bot>: source_documents")
            for document in response["context"]:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
        return answer

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"
