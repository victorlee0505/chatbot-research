import logging
import os
import sys
import time
from typing import Dict, Union, Any, List

import ctransformers
import numpy as np
import torch
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain, LLMChain
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
from hf_llm_config import (
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
from hf_prompts import NO_MEM_PROMPT

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
class HuggingFaceChatBotBase:
    # initialize
    def __init__(
        self,
        llm_config: LLMConfig = None,
        show_callback: bool = False,
        disable_mem: bool = False,
        gpu_layers: int = 0,
        gpu: bool = False,
        server_mode: bool = False,
        log_to_file: bool = False,
    ):
        self.llm_config = llm_config
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
        if self.llm_config.model_type is None:
            torch.set_num_threads(os.cpu_count())
            if not self.gpu:
                self.logger.info("Disable CUDA")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                self.device=torch.device('cpu')
            else:
                self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            self.initialize_model()
        else:
            self.initialize_gguf_model()
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
        try:
            generation_config = GenerationConfig.from_pretrained(self.llm_config.model)
        except Exception as e:
            generation_config = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model, model_max_length=self.llm_config.model_max_length)
        if self.server_mode:
            self.streamer = TextIteratorStreamer(self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
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
        self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

        if self.disable_mem:
            self.qa = LLMChain(llm=self.llm, prompt=self.llm_config.prompt_no_mem_template, verbose=False)
        else:
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.llm_config.max_mem_tokens,
                output_key="response",
                memory_key="history",
                ai_prefix=self.llm_config.ai_prefix,
                human_prefix=self.llm_config.human_prefix,
            )

            self.qa = ConversationChain(llm=self.llm, memory=memory, prompt=self.llm_config.prompt_template, verbose=False)

    def initialize_gguf_model(self):
        self.logger.info("Initializing Model ...")

        model = ctransformers.AutoModelForCausalLM.from_pretrained(self.llm_config.model, 
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
            gpu_layers=self.gpu_layers
            )
        self.tokenizer = ctransformers.AutoTokenizer.from_pretrained(model)

        if self.server_mode:
            self.streamer = TextIteratorStreamer(self.tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
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
        self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

        if self.disable_mem:
            self.qa = LLMChain(llm=self.llm, prompt=self.llm_config.prompt_no_mem_template, verbose=False)
        else:
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.llm_config.max_mem_tokens,
                output_key="response",
                memory_key="history",
                ai_prefix=self.llm_config.ai_prefix,
                human_prefix=self.llm_config.human_prefix,
            )

            self.qa = ConversationChain(llm=self.llm, memory=memory, prompt=self.llm_config.prompt_template, verbose=False)

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
        elif text.lower().strip() in ["reset"]:
            self.logger.info("<bot>: reset conversation memory detected.")
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.llm_config.max_mem_tokens,
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
        if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.server_mode:
            # a closing comment
            answer = "<bot>: See you soon! Bye!"
            print(f"<bot>: {answer}")
            return answer
        if self.inputs.lower().strip() in ["reset"]:
            # a closing comment
            answer = "<bot>: Conversation Memory cleared!"
            print(f"<bot>: {answer}")
            return answer
        response = self.qa({"input": self.inputs})
        if self.disable_mem:
            output_key = "text"
        else:
            output_key = "response"
        answer = (
            response[output_key]
        )
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
    # bot = HuggingFaceChatBotBase(llm_config=REDPAJAMA_3B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=REDPAJAMA_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=VICUNA_7B, disable_mem=True)

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_LONGCHAT_1_5_32K_7B)

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B, disable_mem=True)
    bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_LONGCHAT_1_5_32K_7B, disable_mem=True)

    # GGUF Quantantized LLM, use less RAM
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_13B_Q6, disable_mem=True, gpu_layers=10) # mem = 18GB
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_13B_Q6, disable_mem=True, gpu_layers=10) # mem = 18GB

    # bot = HuggingFaceChatBotBase(llm_config=STARCHAT_BETA_16B_Q5, disable_mem=True, gpu_layers=10) # mem = 23GB

    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_3B, disable_mem=True, gpu_layers=10)
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_15B_Q8, disable_mem=True, gpu_layers=10) # mem = 23GB
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_7B, disable_mem=True, gpu_layers=10)
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_7B_Q6, disable_mem=True, gpu_layers=10) # mem = 9GB
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_13B_Q6, disable_mem=True, gpu_layers=10) # mem = 14GB
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDCODER_PY_34B_Q5, disable_mem=True, gpu_layers=10) # mem = 27GB
    
    # This one is not good at all
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDLM_FALCON_40B_Q6K, disable_mem=True, gpu_layers=10) # mem = 45GB

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
