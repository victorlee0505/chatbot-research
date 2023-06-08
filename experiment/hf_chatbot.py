import logging
import os
import sys
import time

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False
stop_words = ["<human>:"]

# checkpoint 
checkpoint = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class HuggingfaceChatBot():
    # initialize
    def __init__(self, 
                checkpoint: str = None
    ):
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, torch_dtype=torch.bfloat16)
        self.chat_history = None
        self.inputs = None

        self.user_input_ids = None
        self.bot_history_ids = None
        self.stop_words_ids = [
            self.tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
            for stop_word in stop_words
        ]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=self.stop_words_ids)])
        self.end_chat = False
        # greet while starting
        self.welcome()
        
    def welcome(self):
        logger.info("Initializing ChatBot ...")
        # some time to get user ready
        time.sleep(2)
        logger.info('Type "bye" or "quit" or "exit" to end chat \n')
        # give time to read what has been printed
        time.sleep(3)
        # Greet and introduce
        greeting = np.random.choice([
            "Welcome, I am ChatBot, here for your kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a ChatBot. Let's chat!"
        ])
        print("<bot>: " + greeting)
        
    def promptWrapper(self, text: str):
        return "<human>: " + text +"\n<bot>: "
    
    def user_input(self):
        # receive input from user
        text = input("<human>: ")
        logger.debug(text)
        # end conversation if user wishes so
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            # turn flag on 
            self.end_chat=True
            # a closing comment
            logger.info('<bot>: See you soon! Bye!')
            time.sleep(1)
            logger.info('\nQuitting ChatBot ...')
        else:
            # continue chat, preprocess input text
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            prompt = self.promptWrapper(text)
            if self.chat_history is not None:
                self.chat_history += prompt
            else:
                # if first entry, initialize bot_input_ids
                self.chat_history = prompt

            self.inputs = self.tokenizer(self.chat_history, return_tensors='pt').to(self.model.device)
            

    def bot_response(self):
        
        # define the new chat_history_ids based on the preceding chats
        # generated a response while limiting the total chat history to 1000 tokens, 
        bot_output = self.model.generate(**self.inputs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True, stopping_criteria=self.stopping_criteria)

        # last ouput tokens from bot
        # response = tokenizer.decode(output[:, self.new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # outputs = self.tokenizer.decode(bot_output[:, self.chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
        token = bot_output.sequences[0, self.inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(token)
        # in case, bot fails to answer
        if response == "":
            response = self.random_response()
        # print bot response
        self.chat_history += f"{response}\n"
        # logger.debug(self.chat_history)
        print(f"<bot>: {response}")

        logger.debug(f"<history> start")
        logger.debug(f"{self.chat_history}")
        logger.debug(f"<history> ended")
        
    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


# build a ChatBot object
bot = HuggingfaceChatBot(checkpoint = checkpoint)
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