import logging
import json
import os
import sys
import time
from typing import Dict, Union, Any, List

import numpy as np
import torch
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain, create_tagging_chain_pydantic
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from langchain.prompts import ChatPromptTemplate

from PersonalDetails import PersonalDetails

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
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
checkpoint = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"

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

stop_words = ["Question:", "<human>:", "Q:", "Human:", "</s>"]

# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class VicunaChatBotForm:
    # initialize
    def __init__(
        self,
        model: str = None,
        show_callback: bool = False,
        gpu: bool = False,
        gui_mode: bool = False,
        user_details: PersonalDetails = None
    ):
        self.model = model
        self.show_callback = show_callback
        self.gpu = gpu 
        self.gui_mode = gui_mode
        self.llm = None
        self.ai_prefix = None
        self.human_prefix = None
        self.qa = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        self.user_details = user_details
        # greet while starting
        if self.model is None or len(self.model) == 0:
            self.model = checkpoint
        if self.user_details is None:
            self.user_details = PersonalDetails(first_name="",
                                last_name="",
                                full_name="",
                                city="",
                                email="",
                                language="")
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
        tokenizer = AutoTokenizer.from_pretrained(self.model, model_max_length=2048)

        if self.gpu:
            model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True)
            # model = self.model.half().cuda()
            torch_dtype = torch.float16
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True)
            torch_dtype = torch.bfloat16

        if self.gpu:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt").to('cuda')["input_ids"].squeeze()
                for stop_word in stop_words
            ]
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )
        else:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
                for stop_word in stop_words
            ]
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stops=stop_words_ids)]
            )
        
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            device=device,
            # device_map="auto",
            do_sample=True,
            torch_dtype=torch_dtype,
            stopping_criteria=stopping_criteria,
            model_kwargs={"offload_folder": "offload"},
        )

        handler = [MyCustomHandler()] if self.show_callback else None
        self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

        # DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        # Current conversation:
        # {history}
        # Human: {input}
        # AI:"""
        # PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

        # instruct_prefix = "instruct"
        # if instruct_prefix.lower() in self.model.lower():
        #     ai_prefix="Q: "
        #     human_prefix="A: "
        # else:

        PROMPT = ChatPromptTemplate.from_template(
            "USER: Below is are some things to ask the user for in a coversation way. you should only ask one question at a time even if you don't get all the info \
            don't ask as a list! Don't greet the user! Don't say Hi.Explain you need to get some info. If the ask_for list is empty then thank them and ask how you can help them \n\n \
            ### ask_for list: {ask_for}\
            ASSISTANT:"
        )

        self.qa = LLMChain(llm=self.llm, prompt=PROMPT, verbose=False)

    def promptWrapper(self, text: str):
        return "<human>: " + text + "\n<bot>: "
    
    def check_what_is_empty(self, user_peronal_details):
        ask_for = []
        # Check if fields are empty
        for field, value in user_peronal_details.dict().items():
            if value in [None, "", 0]:  # You can add other 'empty' conditions as per your requirements
                print(f"Field '{field}' is empty.")
                ask_for.append(f'{field}')
        return ask_for

    def add_non_empty_details(self, current_details: PersonalDetails, new_details: PersonalDetails):
        non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
        updated_details = current_details.copy(update=non_empty_details)
        return updated_details
    
    def filter_response(self, text_input, user_details):
        chain = create_tagging_chain_pydantic(PersonalDetails, self.llm)
        res = chain.run(text_input)
        # add filtered info to the
        user_details = self.add_non_empty_details(user_details,res)
        ask_for = self.check_what_is_empty(user_details)
        return user_details, ask_for
    
    def ask_for_info(self, ask_for = ['name']):
        ai_chat = self.qa.run(ask_for=ask_for)
        return ai_chat
    
    def generate_form(self):
        file_path = '/output/user_details_Form.json'
        with open(file_path, 'w') as f:
                json.dump(self.user_details, f)

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
        new_user_details, ask_for = self.filter_response(self.inputs, self.user_details)
        self.user_details = new_user_details
        if ask_for:
            ai_response = self.ask_for_info(ask_for)
        else:
            print('Everything gathered, generating form.\n')
            self.generate_form()
            self.end_chat = True
            ai_response = "Form generated, goodbye!"
        # in case, bot fails to answer
        if ai_response == "":
            ai_response = self.random_response()
        else:
            ai_response = ai_response.replace("\n<human>:", "") #chat
            ai_response = ai_response.replace("\nUser", "") #instruct
        # print bot response
        self.chat_history.append((f"<human>: {self.inputs}", f"<bot>: {ai_response}"))
        # logger.info(self.chat_history)
        print(f"<bot>: {ai_response}")
        return ai_response

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":
    # build a ChatBot object
    bot = VicunaChatBotForm()
    # start chatting
    first_prompt = bot.ask_for_info()
    print(f"<bot>: {first_prompt}")
    while True:
        # receive user input
        bot.user_input()
        # check whether to end chat
        if bot.end_chat:
            break
        # output bot response
        bot.bot_response()
        # check whether to end chat
        if bot.end_chat:
            break
    # Happy Chatting!
