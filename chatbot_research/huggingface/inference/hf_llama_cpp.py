import logging
import os

import torch
from ctransformers import AutoModelForCausalLM as cAutoModelForCausalLM
from ctransformers import AutoTokenizer as cAutoTokenizer
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain, LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationSummaryBufferMemory
from transformers import (
    StoppingCriteriaList,
    TextIteratorStreamer,
    TextStreamer,
    pipeline,
)

from chatbot_research.huggingface.config.hf_llm_config import LLMConfig
from chatbot_research.huggingface.inference.hf_custom_callback import MyCustomHandler
from chatbot_research.huggingface.inference.hf_interference_interface import (
    HFInferenceInterface,
)
from chatbot_research.huggingface.inference.hf_stopping_criteria import (
    StoppingCriteriaSub,
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

        if self.logger is None:
            self.logger: logging.Logger = logging.getLogger("HFTransformer")
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

    def initialize_gguf_model(self):
        self.logger.info("Initializing Model ...")

        model = cAutoModelForCausalLM.from_pretrained(
            self.llm_config.model,
            model_file=self.llm_config.model_file,
            model_type=self.llm_config.model_type,
            hf=True,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p,
            top_k=self.llm_config.top_k,
            repetition_penalty=1.2,
            context_length=self.llm_config.model_max_length,
            max_new_tokens=self.llm_config.max_new_tokens,
            # stop=self.llm_config.stop_words,
            threads=os.cpu_count(),
            stream=True,
            gpu_layers=self.gpu_layers,
        )
        self.tokenizer = cAutoTokenizer.from_pretrained(model)

        self.tokenizer = self.llm.tokenizer

        if self.server_mode:
            self.streamer = TextIteratorStreamer(
                self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True
            )
        else:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        stop_words_ids = [
            self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in self.llm_config.stop_words
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.llm_config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            # eos_token_id=self.tokenizer.convert_tokens_to_ids(self.llm_config.eos_token_id) if self.llm_config.eos_token_id is not None else None,
            stopping_criteria=stopping_criteria,
            streamer=self.streamer,
            model_kwargs={"offload_folder": "offload"},
        )
        handler = []
        handler = handler.append(MyCustomHandler()) if self.show_callback else handler
        self.llm_pipeline = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

        if self.disable_mem:
            self.qa = LLMChain(
                llm=self.llm_pipeline,
                prompt=self.llm_config.prompt_no_mem_template,
                verbose=False,
            )
        else:
            memory = ConversationSummaryBufferMemory(
                llm=self.llm_pipeline,
                max_token_limit=self.llm_config.max_mem_tokens,
                output_key="response",
                memory_key="history",
                ai_prefix=self.llm_config.ai_prefix,
                human_prefix=self.llm_config.human_prefix,
            )

            self.qa = ConversationChain(
                llm=self.llm_pipeline,
                memory=memory,
                prompt=self.llm_config.prompt_template,
                verbose=False,
            )
