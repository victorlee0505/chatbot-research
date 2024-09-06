import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory

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

    @abstractmethod
    def reset_chat_memory(self) -> ConversationSummaryBufferMemory:
        raise NotImplementedError("You must implement the reset_chat_memory method.")
