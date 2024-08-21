import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from langchain.memory import ConversationSummaryBufferMemory

from chatbot_research.huggingface.config.hf_llm_config import LLMConfig


class HFInferenceInterface(ABC):
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

    @abstractmethod
    def initialize_torch_device(self) -> torch.device | None:
        raise NotImplementedError(
            "You must implement the initialize_torch_device method."
        )

    @abstractmethod
    def initialize_model(self) -> Any:
        raise NotImplementedError("You must implement the initialize_model method.")

    def initializa_chat_memory(self) -> ConversationSummaryBufferMemory:
        if self.llm_pipeline is None:
            raise ValueError("You must initialize the model first.")
        if self.llm_config is None:
            raise ValueError("You must initialize the config first.")
        return ConversationSummaryBufferMemory(
            llm=self.llm_pipeline,
            max_token_limit=self.llm_config.max_mem_tokens,
            output_key="response",
            memory_key="history",
            ai_prefix=self.llm_config.ai_prefix,
            human_prefix=self.llm_config.human_prefix,
        )
