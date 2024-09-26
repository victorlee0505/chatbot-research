import logging
import os
import time

import numpy as np

from chatbot_research.huggingface.config.hf_llm_config import (
    OPENORCA_MISTRAL_7B_Q5,
    OPENORCA_MISTRAL_8K_7B,
    LLMConfig,
)
from chatbot_research.huggingface.inference.hf_llama_cpp import HFllamaCpp
from chatbot_research.huggingface.inference.hf_transformer import HFTransformer


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
        self.inference = None

        self.logger = logging.getLogger("chatbot-base")
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
        if self.llm_config.model_type is None:
            self.inference = HFTransformer(
                logger=self.logger,
                llm_config=self.llm_config,
                gpu=self.gpu,
                server_mode=self.server_mode,
                disable_mem=self.disable_mem,
                show_callback=self.show_callback,
            )
            self.qa = self.inference.initialize_chain()
        else:
            self.inference = HFllamaCpp(
                logger=self.logger,
                llm_config=self.llm_config,
                gpu=self.gpu,
                server_mode=self.server_mode,
                disable_mem=self.disable_mem,
                show_callback=self.show_callback,
            )
            self.qa = self.inference.initialize_chain()
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
            self.qa = self.inference.reset_chat_memory()
            answer = "<bot>: Conversation Memory cleared!"
            print(f"<bot>: {answer}")
            return answer
        answer = ""
        previous_length = 0
        # Stream the response
        for chunk in self.qa.stream(
            {"input": self.inputs},
            {"configurable": {"session_id": "unused"}},
        ):
            answer += chunk
            if previous_length == 0:
                print(f"<bot>: {chunk}", end="", flush=True)
            else:
                print(chunk, end="", flush=True)  # Print incrementally
            previous_length = len(answer)

        # in case, bot fails to answer
        if answer.strip() == "":
            answer = self.random_response()
            print(f"<bot>: {answer}")
        else:
            answer = answer.replace("\n<human>:", "").replace("\nHuman:", "")
        # print bot response
        self.chat_history.append((f"<human>: {self.inputs}", f"<bot>: {answer}"))
        # logger.info(self.chat_history)

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

    bot = HuggingFaceChatBotBase(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)
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
