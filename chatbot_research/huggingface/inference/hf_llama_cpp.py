import logging
import os

import torch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from llama_cpp import Llama

from chatbot_research.huggingface.config.hf_llm_config import LLMConfig
from chatbot_research.huggingface.config.hf_prompts import CONDENSE_QUESTION_PROMPT
from chatbot_research.huggingface.inference.hf_interference_interface import (
    HFInferenceInterface,
)


class HFllamaCpp(HFInferenceInterface):
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

    def initialize_chain(self):  # -> RunnableSerializable[Dict, str] | Any:
        self.logger.info("Initializing Model ...")
        if self.llm_pipeline is None:
            self.llm_pipeline = self.create_pipeline()

        if self.disable_mem:
            self.qa = self.llm_config.prompt_no_mem_template | self.llm_pipeline
        else:

            chat_history_for_chain = InMemoryChatMessageHistory()
            chain = self.llm_config.prompt_template | self.llm_pipeline
            self.qa = RunnableWithMessageHistory(
                chain,
                lambda session_id: chat_history_for_chain,
                input_messages_key="input",
                history_messages_key="history",
            )

        return self.qa

    def initialize_retrival_chain(self, vectorstore: VectorStore):
        self.logger.info("Initializing Model ...")
        if self.llm_pipeline is None:
            self.llm_pipeline = self.create_pipeline()

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.llm_config.target_source_chunks},
            max_tokens_limit=self.llm_config.retriever_max_tokens_limit,
        )

        if self.disable_mem:
            qa_chain = create_stuff_documents_chain(self.llm_pipeline, self.llm_config.prompt_qa_template)
            self.qa = create_retrieval_chain(retriever, qa_chain)
        else:
            history_aware_retriever = create_history_aware_retriever(
                self.llm_pipeline, retriever, CONDENSE_QUESTION_PROMPT
            )
            qa_chain = create_stuff_documents_chain(
                self.llm_pipeline, self.llm_config.prompt_qa_template
            )
            self.qa = create_retrieval_chain(history_aware_retriever, qa_chain)

        return self.qa

    def create_pipeline(self):
        self.logger.info("Initializing LLM ...")
        self.llm: Llama = Llama.from_pretrained(
            repo_id=self.llm_config.model,
            filename=self.llm_config.model_file,
            verbose=False,
        )

        return LlamaCpp(
            model_path=self.llm.model_path,
            max_tokens=self.llm_config.max_new_tokens,  # Adjust max tokens as needed
            verbose=False,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p,
            top_k=self.llm_config.top_k,
            repeat_penalty=1.2,
            n_ctx=self.llm_config.model_max_length,
            n_batch=self.llm_config.max_new_tokens,
            stop=self.llm_config.stop_words,
            n_threads=os.cpu_count(),
            streaming=True,
        )

    def reset_chat_memory(self):
        chat_history_for_chain = InMemoryChatMessageHistory()
        chain = self.llm_config.prompt_template | self.llm_pipeline
        self.qa = RunnableWithMessageHistory(
            chain,
            lambda session_id: chat_history_for_chain,
            input_messages_key="input",
            history_messages_key="history",
        )
        return self.qa
