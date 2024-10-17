import json
import logging
import os
import time
from typing import Union

from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from chatbot_research.huggingface.config.hf_llm_config import LLMConfig
from chatbot_research.huggingface.inference.hf_chat_llama_cpp import HFChatllamaCpp


# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class HuggingFaceChatBotForm:
    # initialize
    def __init__(
        self,
        llm_config: LLMConfig = None,
        pydantic_model: BaseModel = None,
        result_filename: str = None,
        show_callback: bool = False,
        disable_mem: bool = False,
        gpu_layers: int = 0,
        gpu: bool = False,
        server_mode: bool = False,
        log_to_file: bool = False,
    ):
        self.llm_config = llm_config
        self.pydantic_model = pydantic_model
        self.pydantic_model_object = None
        self.result_filename = result_filename
        self.structured_llm = None
        self.show_callback = show_callback
        self.disable_mem = disable_mem
        self.gpu_layers = gpu_layers
        self.gpu = gpu
        self.device = None
        self.server_mode = server_mode
        self.llm = None
        self.tokenizer = None
        self.streamer = None
        self.qa = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        self.log_to_file = log_to_file
        self.inference = None

        if self.pydantic_model:
            self.dict_schema: dict = convert_to_openai_tool(self.pydantic_model)
        else:
            exit("Please provide a pydantic model for the form")

        fields = {field: "" for field in self.pydantic_model.model_fields}
        self.pydantic_model_object = self.pydantic_model.model_construct(**fields)

        self.logger = logging.getLogger("chatbot-chat-form")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
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
        self.inference = HFChatllamaCpp(
            logger=self.logger,
            llm_config=self.llm_config,
            gpu=self.gpu,
            server_mode=self.server_mode,
            disable_mem=self.disable_mem,
            show_callback=self.show_callback,
        )

        ASK_PROMPT = ChatPromptTemplate.from_template(
            """
            <|im_start|>system
            You are assisting the user in filling out a form. Please be concise.
            <|im_end|>
            <|im_start|>user
            Below is are some things to ask the user for in a coversation way. \
            You should only ask one question for one information at a time even if you don't get all the info. \
            Don't ask as a list! Don't greet the user! Don't say Hi. Explain you need to get some info. \
            If the ask_for list is empty then say "Thank you." \n\n \
            ### ask_for :{ask_for}
            <|im_end|>
            <|im_start|>assistant 
            """
        )

        EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
            """
            <|im_start|>system
            You are expert on extract information. Please be polite and concise.
            <|im_end|>
            <|im_start|>user
            Only extract relevant information from the text. If you do not know the value of an attribute then return empty string for the attribute's value. \
            ### info : {info}
            <|im_end|>
            <|im_start|>assistant 
            """
        )
        self.llm = self.inference.initialize_chat_llm(temperature=0)
        self.qa = self.inference.initialize_chat_model(ASK_PROMPT)
        self.structured_llm = EXTRACTION_PROMPT | self.llm.with_structured_output(
            self.dict_schema
        )

        # some time to get user ready
        time.sleep(2)
        self.logger.info('Type "bye" or "quit" or "exit" to end chat \n')

    def filter_response(self, text_input, pydantic_object: BaseModel):

        result = self.structured_llm.invoke({"info": text_input})
        self.logger.info(f"new result : {result}")
        # add filtered info to the
        pydantic_object = self.add_non_empty_details(pydantic_object, result)
        ask_for = self.check_what_is_empty(pydantic_object)
        return pydantic_object, ask_for

    def add_non_empty_details(
        self, current_details: BaseModel, new_details: Union[BaseModel, dict]
    ):

        if isinstance(new_details, BaseModel):
            new_details_dict = new_details.model_dump()
        else:
            new_details_dict = new_details  # Assume it's already a dict

        self.logger.info(f"new_details_dict: {new_details_dict}")
        non_empty_details = {
            k: v for k, v in new_details_dict.items() if v not in [None, ""]
        }
        self.logger.info(f"non_empty_details: {non_empty_details}")
        updated_details = current_details.model_copy(update=non_empty_details)

        return updated_details

    def check_what_is_empty(self, pydantic_object: BaseModel):
        ask_for = []
        # Check if fields are empty
        for field, value in pydantic_object.model_dump().items():
            if value in [
                None,
                "",
                0,
                "null",
            ]:  # You can add other 'empty' conditions as per your requirements
                self.logger.info(f"Field '{field}' is empty.")
                ask_for.append(f"{field}")
        return ask_for

    def ask_for_info(self, ask_for: list = None):
        if ask_for is None:
            # first key of self.dict_schema
            self.logger.info(
                f"Asking for {self.dict_schema['function']['parameters']['properties'].keys()}"
            )
            ask_for = list(
                self.dict_schema["function"]["parameters"]["properties"].keys()
            )
        self.logger.info(f"Ask for: {ask_for}")
        # ai_chat = self.structured_llm.invoke(input=f'generate a question to ask user for this information: {str(ask_for)}')
        ai_chat = self.qa.invoke({"ask_for": ask_for})
        return ai_chat

    def generate_form(self):
        if self.result_filename is None:
            self.result_filename = "result.json"

        output_dir = "/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, self.result_filename)
        with open(file_path, "w") as f:
            json.dump(self.pydantic_model_object, f)

    def process_info(self):
        first_prompt = self.ask_for_info().content
        print(f"<bot>: {first_prompt}")
        while True:
            # receive user input
            text = input("<human>: ")
            if text.lower().strip() in ["bye", "quit", "exit"]:
                self.end_chat = True
                # check whether to end chat
                answer = "See you soon! Bye!"
                print(f"<bot>: {answer}")
            if self.end_chat:
                break
            # output bot response
            new_pydantic_object, ask_for = self.filter_response(
                text_input=text, pydantic_object=self.pydantic_model_object
            )
            self.pydantic_model_object = new_pydantic_object
            if ask_for:
                follow_up_prompt = self.ask_for_info(ask_for).content
                print(f"<bot>: {follow_up_prompt}")
            else:
                print("Everything gathered, generating form.\n")
                self.generate_form()
                self.end_chat = True
                answer = "Form generated, goodbye!"
                print(f"<bot>: {answer}")
                # check whether to end chat
                if self.end_chat:
                    break
