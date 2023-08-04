import logging
import os
import sys
import time
from typing import Dict, Union, Any, List

import numpy as np
import torch
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

from hf_llm_config import CODEGEN25_7B, CODEGEN2_1B, CODEGEN2_4B, SANTA_CODER_1B, LLMConfig

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
        gui_mode: bool = False,
        log_to_file: bool = False,
    ):
        self.llm_config = llm_config
        self.tokenizer = None
        self.model = None
        self.gpu = gpu
        self.device = None
        self.gui_mode = gui_mode
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model, model_max_length=self.llm_config.model_max_length, trust_remote_code=True)

        if self.gpu:
            self.model = AutoModelForCausalLM.from_pretrained(self.llm_config.model, trust_remote_code=True).to(self.device)
            self.torch_dtype = torch.float16
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.llm_config.model, trust_remote_code=True).to(self.device)
            self.torch_dtype = torch.bfloat16

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
                output_key="response",
                memory_key="history",
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
        inputs = self.tokenizer.encode(self.inputs, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=400)
        answer = self.tokenizer.decode(outputs[0])
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
    bot = HuggingFaceChatBotCoder(llm_config=SANTA_CODER_1B)
    # bot = HuggingFaceChatBotCoder(llm_config=CODEGEN2_1B)
    # bot = HuggingFaceChatBotCoder(llm_config=CODEGEN2_4B)
    # bot = HuggingFaceChatBotCoder(llm_config=CODEGEN25_7B)

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
