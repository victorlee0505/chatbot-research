import logging
import json
import os
import sys
import time
from typing import Dict, Union, Any, List

from ctransformers import (
    AutoModelForCausalLM as cAutoModelForCausalLM,
    AutoTokenizer as cAutoTokenizer,
)

import numpy as np
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain, LLMChain, create_tagging_chain_pydantic, create_tagging_chain, create_extraction_chain, create_extraction_chain_pydantic
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationSummaryBufferMemory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
    TextIteratorStreamer,
    pipeline,
)
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from chatbot_research.huggingface.chatbots.hf_chatbot_base import HuggingFaceChatBotBase
from chatbot_research.huggingface.config.hf_llm import HuggingFaceLLM
from chatbot_research.huggingface.config.hf_llm_config import (
    REDPAJAMA_3B,
    REDPAJAMA_7B,
    VICUNA_7B,
    LMSYS_VICUNA_1_5_7B,
    LMSYS_VICUNA_1_5_16K_7B,
    LMSYS_LONGCHAT_1_5_32K_7B,
    LMSYS_VICUNA_1_5_7B_Q8,
    LMSYS_VICUNA_1_5_16K_7B_Q8,
    LMSYS_VICUNA_1_5_13B_Q6,
    LMSYS_VICUNA_1_5_16K_13B_Q6,
    OPENORCA_MISTRAL_8K_7B,
    OPENORCA_MISTRAL_7B_Q5,
    STARCHAT_BETA_16B_Q5,
    WIZARDCODER_3B,
    WIZARDCODER_15B_Q8,
    WIZARDCODER_PY_7B,
    WIZARDCODER_PY_7B_Q6,
    WIZARDCODER_PY_13B_Q6,
    WIZARDCODER_PY_34B_Q5,
    WIZARDLM_FALCON_40B_Q6K, 
    LLMConfig
)
from chatbot_research.huggingface.chat_models.hf_base_chat_model import ChatHuggingFace
from chatbot_research.form_model.personal_details import PersonalDetails

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
class HuggingFaceChat2Form:
    # initialize
    def __init__(
        self,
        llm : HuggingFaceLLM = None,
        user_details : BaseModel = None
    ):
        self.llm : HuggingFaceLLM = llm
        self.user_details : BaseModel = user_details
        self.chat : ChatHuggingFace = None
        self.end_chat = False

        if self.user_details is None:
            self.user_details = PersonalDetails(first_name="",
                                last_name="",
                                full_name="",
                                city="",    
                                email="",
                                language="")
        # greet while starting
        self.welcome()

    def welcome(self):

        self.chat = self.create_chat_model(llm=self.llm)

        PROMPT = ChatPromptTemplate.from_template(
            """
            <|im_start|>user
            Below is are some things to ask the user for in a coversation way. you should only ask one question at a time even if you don't get all the info \
            don't ask as a list! Don't greet the user! Don't say Hi.Explain you need to get some info. If the ask_for list is empty then thank them and ask how you can help them \n\n \
            ### ask_for list:{ask_for}<|im_end|>
            <|im_start|>assistant 
            """
        )
        # PROMPT = ChatPromptTemplate.from_template(
        #     "Below is are some things to ask the user for in a coversation way. you should only ask one question at a time even if you don't get all the info \
        #     don't ask as a list! Don't greet the user! Don't say Hi.Explain you need to get some info. If the ask_for list is empty then thank them and ask how you can help them \n\n \
        #     ### ask_for list: {ask_for}"
        # )

        self.qa = LLMChain(llm=self.chat, prompt=PROMPT, verbose=False)
        print("Welcome to the chatbot, please enter your details below.")

    def create_chat_model(self, llm):
        """
        Create a chat model with the specified model name and temperature.

        Parameters
        ----------
        model : str
            The name of the model to create.
        temperature : float
            The temperature to use for the model.

        Returns
        -------
        BaseChatModel
            The created chat model.
        """
        return ChatHuggingFace(
            llm=llm,
        )

    def check_what_is_empty(self, user_peronal_details: BaseModel):
        ask_for = []
        # Check if fields are empty
        for field, value in user_peronal_details.dict().items():
            if value in [None, "", 0]:  # You can add other 'empty' conditions as per your requirements
                print(f"Field '{field}' is empty.")
                ask_for.append(f'{field}')
        return ask_for

    def add_non_empty_details(self, current_details: BaseModel, new_details: BaseModel):
        non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
        updated_details = current_details.copy(update=non_empty_details)
        return updated_details
    
    # def _get_tagging_function(self, schema: dict) -> dict:
    #     return {
    #         "name": "information_extraction",
    #         "description": "Extracts the relevant information from the passage.",
    #         "parameters": self._convert_schema(schema),
    #     }

    # def _convert_schema(self, schema: dict) -> dict:
    #     props = {k: {"title": k, **v} for k, v in schema["properties"].items()}
    #     return {
    #         "type": "object",
    #         "properties": props,
    #         "required": schema.get("required", []),
    #     }

    def filter_response(self, text_input, user_details: BaseModel):

        schema = {
            "properties": {
                'first_name': {'title': 'First Name', 'description': 'This is the first name of the user.', 'type': 'string'}, 
                'last_name': {'title': 'Last Name', 'description': 'This is the last name or surname of the user.', 'type': 'string'}, 
                'full_name': {'title': 'Full Name', 'description': 'Is the full name of the user ', 'type': 'string'}, 
                'city': {'title': 'City', 'description': 'The name of the city where someone lives', 'type': 'string'}, 
                'email': {'title': 'Email', 'description': 'an email address that the person associates as theirs', 'type': 'string'}, 
                'language': {'title': 'Language', 'enum': ['spanish', 'english', 'french', 'german', 'italian'], 'type': 'string'},
            },
            "required": ["first_name", "last_name", "full_name", "city", "email", "language"],
        }
        openai_schema = PersonalDetails.schema()
        # function = self._get_tagging_function(openai_schema)
        # chain = create_tagging_chain(schema=schema, llm=self.chat)
        chain = create_tagging_chain_pydantic(pydantic_schema=PersonalDetails, llm=self.chat)
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

    def process_info(self):
        first_prompt = self.ask_for_info()
        print(f"<bot>: {first_prompt}")
        while True:
            # receive user input
            text = input("<human>: ")
            if text.lower().strip() in ["bye", "quit", "exit"]:
                self.end_chat = True
            # check whether to end chat
            if self.end_chat:
                break
            # output bot response
            new_user_details, ask_for = self.filter_response(text_input=text, user_details=self.user_details)
            self.user_details = new_user_details
            if ask_for:
                answer = self.ask_for_info(ask_for)
            else:
                print('Everything gathered, generating form.\n')
                self.generate_form()
                self.end_chat = True
                answer = "Form generated, goodbye!"
                print(f"<bot>: {answer}")
                # check whether to end chat
                if self.end_chat:
                    break
    # Happy Chatting!

    # def user_input(self, prompt: str = None):
    #     # receive input from user
    #     if prompt:
    #         text = prompt
    #     else:
    #         text = input("<human>: ")

    #     # end conversation if user wishes so
    #     if text.lower().strip() in ["bye", "quit", "exit"] and not self.server_mode:
    #         # turn flag on
    #         self.end_chat = True
    #         # a closing comment
    #         print("<bot>: See you soon! Bye!")
    #         time.sleep(1)
    #         self.logger.info("\nQuitting ChatBot ...")
    #         self.inputs = text
    #     else:
    #         self.inputs = text

    # @timer_decorator
    # def bot_response(self) -> str:
    #     if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.server_mode:
    #         # a closing comment
    #         answer = "<bot>: See you soon! Bye!"
    #         print(f"<bot>: {answer}")
    #         return answer
    #     new_user_details, ask_for = self.filter_response(text_input=self.inputs, user_details=self.user_details)
    #     self.user_details = new_user_details
    #     if ask_for:
    #         answer = self.ask_for_info(ask_for)
    #     else:
    #         print('Everything gathered, generating form.\n')
    #         self.generate_form()
    #         self.end_chat = True
    #         answer = "Form generated, goodbye!"
    #     # in case, bot fails to answer
    #     if answer == "":
    #         answer = self.random_response()
    #     # print bot response
    #     self.chat_history.append((self.inputs, answer))
    #     # logger.debug(self.chat_history)
    #     print(f"<bot>: {answer}")
    #     self.count_tokens(self.qa, self.inputs)
    #     return answer

    # # in case there is no response from model
    # def random_response(self):
    #     return "I don't know", "I am not sure"


if __name__ == "__main__":

    # get config
    # build a ChatBot object
    # bot = HuggingFaceChatBotBase(llm_config=REDPAJAMA_3B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=REDPAJAMA_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=VICUNA_7B, disable_mem=True)

    # bot = HuggingFaceChatBotBase(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=OPENORCA_MISTRAL_7B_Q5, disable_mem=True, gpu=False, gpu_layers=0)

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_LONGCHAT_1_5_32K_7B)

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_LONGCHAT_1_5_32K_7B, disable_mem=True)

    # GGUF Quantantized LLM, use less RAM
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_13B_Q6, disable_mem=True, gpu_layers=10) # mem = 18GB
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_13B_Q6, disable_mem=True, gpu_layers=0) # mem = 18GB

    # bot = HuggingFaceChatBotBase(llm_config=STARCHAT_BETA_16B_Q5, disable_mem=True, gpu_layers=0) # mem = 23GB

    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_3B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_15B_Q8, disable_mem=True, gpu_layers=10) # mem = 23GB
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_7B, disable_mem=True, gpu_layers=10)
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_7B_Q6, disable_mem=True, gpu_layers=10) # mem = 9GB
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_13B_Q6, disable_mem=True, gpu_layers=10) # mem = 14GB
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_34B_Q5, disable_mem=True, gpu_layers=10) # mem = 27GB
    
    # This one is not good at all
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDLM_FALCON_40B_Q6K, disable_mem=True, gpu_layers=10) # mem = 45GB

    llm = HuggingFaceLLM(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)

    # start chatting
    processor = HuggingFaceChat2Form(llm=llm)
    processor.process_info()