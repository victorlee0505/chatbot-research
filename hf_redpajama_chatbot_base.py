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

# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 50
# set maximum chunk overlap
max_chunk_overlap = 20

# checkpoint
checkpoint = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False

stop_words = ["Question:", "<human>:", "<bot>:"]

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class RedpajamaChatBotBase:
    # initialize
    def __init__(
        self,
        model: str = None,
        gpu: bool = False,
        gui_mode: bool = False,
    ):
        self.model = model
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
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        if self.gpu:
            model = AutoModelForCausalLM.from_pretrained(checkpoint)
            model = self.model.half().cuda()
            torch_dtype = torch.float16
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint)
            torch_dtype = torch.bfloat16

        stop_words_ids = [
            tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in stop_words
        ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto",
            do_sample=True,
            torch_dtype=torch_dtype,
            stopping_criteria=stopping_criteria,
            model_kwargs={"offload_folder": "offload"},
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        # Current conversation:
        # {history}
        # Human: {input}
        # AI:"""
        # PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            ai_prefix="<bot>: ",
            human_prefix="<human>: ",
        )
        self.qa = ConversationChain(llm=self.llm, memory=memory, verbose=False)

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
        answer = self.qa.run(self.promptWrapper(self.inputs))
        # in case, bot fails to answer
        if answer == "":
            answer = self.random_response()
        else:
            answer = answer.replace("\n<human>:", "")
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
    bot = RedpajamaChatBotBase()
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
