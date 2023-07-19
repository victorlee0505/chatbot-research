import logging
import os
import sys
import time

import numpy as np
import torch
from langchain import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
# handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

# checkpoint
checkpoint = "bigcode/santacoder"

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False

stop_words = ["Question:", "<human>:", "Q:", "Human:"]

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class SantaCoderChatBotBase:
    # initialize
    def __init__(
        self,
        model: str = None,
        gpu: bool = False,
        gui_mode: bool = False,
    ):
        self.model = model
        self.tokenizer = None
        self.device = None
        self.torch_dtype = None
        self.gpu = gpu 
        self.gui_mode = gui_mode
        self.llm = None
        self.qa = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        # greet while starting
        if self.model is None or len(self.model) == 0:
            self.model = checkpoint
        self.welcome()

    def welcome(self):
        logger.info("Initializing ChatBot ...")
        torch.set_num_threads(os.cpu_count())
        if not self.gpu:
            logger.info("Disable CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.initialize_model()
        # some time to get user ready
        time.sleep(2)
        logger.info('Type "bye" or "quit" or "exit" to end chat \n')
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
        logger.info("Initializing Model ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        if self.gpu:
            self.model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True)
            # model = self.model.half().cuda()
            self.torch_dtype = torch.float16
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True)
            self.torch_dtype = torch.bfloat16
        
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        

    def promptWrapper(self, text: str):
        return "<human>: " + text + "\n<bot>: "

    def user_input(self, prompt: str = None):
        # receive input from user
        if prompt:
            text = prompt
        else:
            text = input("<human>: ")

        logger.debug(text)
        # end conversation if user wishes so
        if text.lower().strip() in ["bye", "quit", "exit"] and not self.gui_mode:
            # turn flag on
            self.end_chat = True
            # a closing comment
            logger.info("<bot>: See you soon! Bye!")
            time.sleep(1)
            logger.info("\nQuitting ChatBot ...")
            self.inputs = text
        else:
            self.inputs = text

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
            answer = answer.replace("\nUser", "") #instruct
        # print bot response
        self.chat_history.append((f"<human>: {self.inputs}", f"<bot>: {answer}"))
        # logger.info(self.chat_history)
        print(f"<bot>: {answer}")
        return answer

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":
    # build a ChatBot object
    bot = SantaCoderChatBotBase()
    # bot = SantaCoderChatBotBase(gpu=True)
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
