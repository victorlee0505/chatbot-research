import logging
import os

import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
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


class HFTransformer(HFInferenceInterface):
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
        torch.set_num_threads(os.cpu_count())
        self.logger.info("Initializing torch device ...")
        if self.gpu:
            self.device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.logger.info("Disable CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device = torch.device("cpu")
        return self.device

    def initialize_model(self):  # -> RunnableSerializable[Dict, str] | Any:

        self.initialize_torch_device()

        self.logger.info("Initializing Model ...")
        try:
            generation_config = GenerationConfig.from_pretrained(self.llm_config.model)
        except Exception as e:
            generation_config = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.model, model_max_length=self.llm_config.model_max_length
        )
        if self.server_mode:
            self.streamer = TextIteratorStreamer(
                self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True
            )
        else:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        if self.gpu:
            model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
            model.half().cuda()
            torch_dtype = torch.float16
        else:
            model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
            torch_dtype = torch.bfloat16

        if self.gpu:
            stop_words_ids = [
                self.tokenizer(stop_word, return_tensors="pt")
                .to("cuda")["input_ids"]
                .squeeze()
                for stop_word in self.llm_config.stop_words
            ]
        else:
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
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p,
            top_k=self.llm_config.top_k,
            generation_config=generation_config,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            device=self.device,
            do_sample=self.llm_config.do_sample,
            torch_dtype=torch_dtype,
            stopping_criteria=stopping_criteria,
            streamer=self.streamer,
            model_kwargs={"offload_folder": "offload"},
        )
        handler = []
        handler = handler.append(MyCustomHandler()) if self.show_callback else handler
        self.llm_pipeline = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

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
