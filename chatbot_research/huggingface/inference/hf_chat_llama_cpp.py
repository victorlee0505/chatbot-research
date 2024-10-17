import logging
import os

import torch
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from llama_cpp import Llama

from chatbot_research.huggingface.chat_models.chat_llamacpp import ChatLlamaCppX
from chatbot_research.huggingface.config.hf_llm_config import LLMConfig


class HFChatllamaCpp:
    def __init__(
        self,
        logger: logging.Logger = None,
        llm_config: LLMConfig = None,
        gpu: bool = False,
        server_mode: bool = False,
        disable_mem: bool = False,
        show_callback: bool = False,
    ):
        self.logger = logger
        self.llm_config = llm_config
        self.gpu = gpu
        self.server_mode = server_mode
        self.disable_mem = disable_mem
        self.show_callback = show_callback
        self.device = None
        self.llm_pipeline = None
        self.llm = None
        self.chat_llm = None
        self.store = {}

        if self.logger is None:
            self.logger: logging.Logger = logging.getLogger("HFllamaCpp")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def initialize_torch_device(self) -> torch.device:
        self.logger.info("Initializing torch device ...")
        torch.set_num_threads(os.cpu_count())
        return None
    
    def initialize_chat_llm(self, temperature: int = None):

        self.llm = self.create_llm()
        if temperature is None:
            temperature = self.llm_config.temperature
        self.chat_llm = ChatLlamaCppX(
            client=self.llm,
            model_path=self.llm_config.model,
            stop=self.llm_config.stop_words,
            temperature=temperature
        )

        return self.chat_llm

    def initialize_chat_model(self, prompt: ChatPromptTemplate, temperature: int = None):

        self.llm = self.create_llm()
        if temperature is None:
            temperature = self.llm_config.temperature
        self.chat_llm = ChatLlamaCppX(
            client=self.llm,
            model_path=self.llm_config.model,
            stop=self.llm_config.stop_words,
            temperature=temperature
        )

        if prompt is not None:
            self.qa = prompt | self.chat_llm
        else:
            if self.disable_mem:
                self.qa = self.llm_config.prompt_no_mem_template | self.chat_llm
            else:

                chat_history_for_chain = InMemoryChatMessageHistory()
                chain = self.llm_config.prompt_template | self.chat_llm
                self.qa = RunnableWithMessageHistory(
                    chain,
                    lambda session_id: chat_history_for_chain,
                    input_messages_key="input",
                    history_messages_key="history",
                )

        return self.qa

    def create_llm(self) -> Llama:
        self.logger.info("Initializing LLM ...")
        if self.llm is None:
            self.llm: Llama = Llama.from_pretrained(
                repo_id=self.llm_config.model,
                filename=self.llm_config.model_file,
                max_tokens=self.llm_config.max_new_tokens,  # Adjust max tokens as needed
                verbose=False,
                # temperature=temperature,
                top_p=self.llm_config.top_p,
                top_k=self.llm_config.top_k,
                repeat_penalty=1.2,
                n_ctx=self.llm_config.model_max_length,
                n_batch=self.llm_config.max_new_tokens,
                stop=self.llm_config.stop_words,
                n_threads=os.cpu_count(),
                streaming=True,
            )

        return self.llm

    def reset_chat_memory(self):
        chat_history_for_chain = InMemoryChatMessageHistory()
        chain = self.llm_config.prompt_template | self.llm
        self.qa = RunnableWithMessageHistory(
            chain,
            lambda session_id: chat_history_for_chain,
            input_messages_key="input",
            history_messages_key="history",
        )
        return self.qa
