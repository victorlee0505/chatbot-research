import logging
import os
import sys
import time
from typing import Dict, Union, Any, List

import numpy as np
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
    TextIteratorStreamer,
    pipeline,
)
from langchain.prompts import PromptTemplate

from hf_llm_config import CODEGEN25_7B, CODEGEN2_1B, CODEGEN2_4B, SANTA_CODER_1B, WIZARDCODER_3B, WIZARDCODER_15B_Q8, WIZARDCODER_PY_7B, LLMConfig

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
class HuggingFaceChatBotCoder:
    # initialize
    def __init__(
        self,
        llm_config: LLMConfig = None,
        gpu: bool = False,
        server_mode: bool = False,
        show_callback: bool = False,
        log_to_file: bool = False,
    ):
        self.llm_config = llm_config
        self.show_callback = show_callback
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.gpu = gpu
        self.device = None
        self.streamer = None
        self.server_mode = server_mode
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        self.log_to_file = log_to_file

        self.logger = logging.getLogger("chatbot-base")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
            self.logger.info("Disable CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device=torch.device('cpu')
        else:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
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

    def initialize_model(self):
        self.logger.info("Initializing Model ...")
        # generation_config = GenerationConfig.from_pretrained(self.llm_config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model, model_max_length=self.llm_config.model_max_length, trust_remote_code=True)
        if self.server_mode:
            self.streamer = TextIteratorStreamer(self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
        else:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        if self.gpu:
            self.model = AutoModelForCausalLM.from_pretrained(self.llm_config.model, trust_remote_code=True).to(self.device)
            self.torch_dtype = torch.float16
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.llm_config.model, trust_remote_code=True).to(self.device)
            self.torch_dtype = torch.bfloat16
        
        if self.gpu:
            stop_words_ids = [
                self.tokenizer(stop_word, return_tensors="pt").to('cuda')["input_ids"].squeeze()
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

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.llm_config.max_new_tokens,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p,
            # top_k=self.llm_config.top_k,
            # generation_config=generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            device=self.device,
            # do_sample=self.llm_config.do_sample,
            torch_dtype=self.torch_dtype,
            # stopping_criteria=stopping_criteria,
            streamer=self.streamer,
            model_kwargs={"offload_folder": "offload"},
        )
        handler = [MyCustomHandler()] if self.show_callback else None
        self.llm = HuggingFacePipeline(pipeline=self.pipe, callbacks=handler)
        PROMPT = PromptTemplate.from_template("{input}")
        self.qa = LLMChain(llm=self.llm, prompt=PROMPT, verbose=False)

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
        else:
            self.inputs = text

    @timer_decorator
    def bot_response(self) -> str:
        if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.server_mode:
            # a closing comment
            answer = "<bot>: See you soon! Bye!"
            print(f"<bot>: {answer}")
            return answer
        
        # inputs = self.tokenizer(self.inputs, return_tensors="pt").to(self.device).input_ids
        # sample = self.model.generate(inputs, max_length=self.llm_config.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        # answer = self.tokenizer.decode(sample[0], skip_special_tokens=True)
        
        # response = self.pipe(self.inputs)
        # answer = response[0]['generated_text']

        answer = self.qa.run(self.inputs)
        
        # in case, bot fails to answer
        if answer == "":
            answer = self.random_response()
        else:
            answer = answer.replace("\n<human>:", "") #chat
            answer = answer.replace("\nHuman:", "") #instruct
        # print bot response
        self.chat_history.append((f"<human>: {self.inputs}", f"<bot>: {answer}"))
        # logger.info(self.chat_history)
        print(f"<bot>: {answer}")
        return answer

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":

    # get config
    # build a ChatBot object

    # bot = HuggingFaceChatBotCoder(llm_config=SANTA_CODER_1B)
    # bot = HuggingFaceChatBotCoder(llm_config=CODEGEN25_7B)
    bot = HuggingFaceChatBotCoder(llm_config=WIZARDCODER_3B)
    # bot = HuggingFaceChatBotCoder(llm_config=WIZARDCODER_PY_7B)

    # These 2 not that good.
    # bot = HuggingFaceChatBotCoder(llm_config=CODEGEN2_1B, gpu=True)
    # bot = HuggingFaceChatBotCoder(llm_config=CODEGEN2_4B, gpu=True)

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
