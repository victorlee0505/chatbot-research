import logging
import os
import sys
import time
from typing import Any, Dict, List, Union

import chromadb
import ctransformers
import numpy as np
import torch
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
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
from file_system import LocalFileStore
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
from hf_prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT_DOCUMENT_CHAT
from ingest_constants import (
    CHROMA_SETTINGS_HF,
    PERSIST_DIRECTORY_HF,
    PERSIST_DIRECTORY_PARENT_HF
    )
from ingest_multi import Ingestion, CHUNK_SIZE

persist_directory = PERSIST_DIRECTORY_HF

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
class HuggingFaceChatBotChroma:
    # initialize
    def __init__(
        self,
        llm_config: LLMConfig = None,
        disable_mem: bool = False,
        load_data: bool = False,
        show_stream: bool = False,
        show_source: bool = False,
        show_callback: bool = False,
        gpu_layers: int = 0,
        gpu: bool = False,
        server_mode: bool = False,
        log_to_file: bool = False,
    ):
        self.llm_config = llm_config
        self.disable_mem = disable_mem
        self.load_data = load_data
        self.show_stream = show_stream
        self.show_source = show_source
        self.show_callback = show_callback
        self.gpu_layers = gpu_layers
        self.gpu = gpu
        self.device = None
        self.server_mode = server_mode
        self.llm = None
        self.streamer = None
        self.embedding_llm = None
        self.qa = None
        self.chat_history = []
        self.inputs = None
        self.end_chat = False
        self.log_to_file = log_to_file

        self.logger = logging.getLogger("chatbot-chroma")
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
            torch.device('cpu')
            self.logger.info("Disable CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device=torch.device('cpu')
        else:
            torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.ingest_documents()

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

    def ingest_documents(self):
        offline = True
        if self.load_data:
            Ingestion(offline=offline)
        else:
            if os.path.exists(persist_directory):
                if os.listdir(persist_directory):
                    self.logger.info(f"Ingestion skipped!")
                else:
                    self.logger.info("PERSIST_DIRECTORY is empty.")
                    Ingestion(offline=offline)
            else:
                self.logger.info("PERSIST_DIRECTORY does not exist.")
                Ingestion(offline=offline)

    def initialize_model(self):
        self.logger.info("Initializing Model ...")
        try:
            generation_config = GenerationConfig.from_pretrained(self.llm_config.model)
        except Exception as e:
            generation_config = None
        if self.gpu:
            self.embedding_llm = HuggingFaceEmbeddings()
        else:
            model_kwargs = {'device': 'cpu'}
            self.embedding_llm = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)
        client = chromadb.PersistentClient(settings=CHROMA_SETTINGS_HF, path=PERSIST_DIRECTORY_HF)
        db = Chroma(
            client=client,
            embedding_function=self.embedding_llm,
        )
        local_store = LocalFileStore(PERSIST_DIRECTORY_PARENT_HF)
        retriever = ParentDocumentRetriever(
            vectorstore=db, 
            docstore=local_store, 
            child_splitter=child_splitter,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model, model_max_length=self.llm_config.model_max_length)
        if self.server_mode:
            self.streamer = TextIteratorStreamer(tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
        else:
            self.streamer = TextStreamer(tokenizer, skip_prompt=True)

        if self.gpu:
            model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
            model.half().cuda()
            torch_dtype = torch.float16
        else:
            model = AutoModelForCausalLM.from_pretrained(self.llm_config.model)
            torch_dtype = torch.bfloat16

        if self.gpu:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt").to("cuda")["input_ids"].squeeze()
                for stop_word in self.llm_config.stop_words
            ]
        else:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
                for stop_word in self.llm_config.stop_words
            ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.llm_config.max_new_tokens,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p,
            top_k=self.llm_config.top_k,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id,
            device=self.device,
            do_sample=self.llm_config.do_sample,
            torch_dtype=torch_dtype,
            stopping_criteria=stopping_criteria,
            streamer=self.streamer,
            model_kwargs={"offload_folder": "offload"},
        )

        handler = []
        handler.append(MyCustomHandler()) if self.show_callback else None
        self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.llm_config.max_mem_tokens,
            output_key="answer",
            memory_key="chat_history",
            ai_prefix=self.llm_config.ai_prefix,
            human_prefix=self.llm_config.human_prefix,
        )

        if self.disable_mem:
            print(f"disable_mem: {self.disable_mem}")
            self.qa = RetrievalQA.from_llm(
                llm=self.llm,
                # chain_type="stuff",
                retriever=retriever,
                # memory=memory,
                prompt=self.llm_config.prompt_qa_template,
                # combine_docs_chain_kwargs={"prompt": QA_PROMPT_DOCUMENT_CHAT},
                # get_chat_history=lambda h: h,
                return_source_documents=self.show_source,
            )
        else:
            print(f"disable_mem: {self.disable_mem}")
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                memory=memory,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                get_chat_history=lambda h: h,
                return_source_documents=self.show_source,
            )
            # still WIP, looks like it still answering outside of the context
            

    def initialize_gguf_model(self):
        self.logger.info("Initializing Model ...")

        if self.gpu:
            self.embedding_llm = HuggingFaceEmbeddings()
        else:
            model_kwargs = {'device': 'cpu'}
            self.embedding_llm = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)
        client = chromadb.PersistentClient(settings=CHROMA_SETTINGS_HF, path=PERSIST_DIRECTORY_HF)
        db = Chroma(
            client=client,
            embedding_function=self.embedding_llm,
        )
        local_store = LocalFileStore(PERSIST_DIRECTORY_PARENT_HF)
        retriever = ParentDocumentRetriever(
            vectorstore=db, 
            docstore=local_store, 
            child_splitter=child_splitter,
        )

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
        tokenizer = ctransformers.AutoTokenizer.from_pretrained(model)

        if self.server_mode:
            self.streamer = TextIteratorStreamer(tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True)
        else:
            self.streamer = TextStreamer(tokenizer, skip_prompt=True)

        stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt")["input_ids"].squeeze()
                for stop_word in self.llm_config.stop_words
            ]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.llm_config.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            # eos_token_id=tokenizer.convert_tokens_to_ids(self.llm_config.eos_token_id) if self.llm_config.eos_token_id is not None else None,
            stopping_criteria=stopping_criteria,
            streamer=self.streamer,
            model_kwargs={"offload_folder": "offload"},
        )
        handler = []
        handler = handler.append(MyCustomHandler()) if self.show_callback else handler
        self.llm = HuggingFacePipeline(pipeline=pipe, callbacks=handler)

        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.llm_config.max_mem_tokens,
            output_key="answer",
            memory_key="chat_history",
            ai_prefix=self.llm_config.ai_prefix,
            human_prefix=self.llm_config.human_prefix,
        )

        if self.disable_mem:
            print(f"disable_mem: {self.disable_mem}")
            self.qa = RetrievalQA.from_llm(
                llm=self.llm,
                # chain_type="stuff",
                retriever=retriever,
                # memory=memory,
                prompt=self.llm_config.prompt_qa_template,
                # chain_type_kwargs={"prompt": self.llm_config.prompt_qa_template},
                # get_chat_history=lambda h: h,
                return_source_documents=self.show_source,
            )
        else:
            print(f"disable_mem: {self.disable_mem}")
            self.qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                memory=memory,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                get_chat_history=lambda h: h,
                return_source_documents=self.show_source,
            )

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
                output_key="answer",
                memory_key="chat_history",
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
        
        if self.disable_mem:
            input_key = "query"
            output_key = "result"
        else:
            input_key = "question"
            output_key = "answer"

        response = self.qa({input_key: self.inputs})
        answer, docs = (
            response[output_key],
            response["source_documents"] if self.show_source else [],
        )

        # in case, bot fails to answer
        if answer == "":
            answer = self.random_response()
        else:
            answer = answer.replace("\n<human>:", "")
        # print bot response
        self.chat_history.append((f"<human>: {self.inputs}", f"<bot>: {answer}"))
        # logger.info(self.chat_history)
        print(f"<bot>: {answer}")
        if self.show_source:
            print(f"<bot>: source_documents")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
        return answer

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":
    # get config
    # build a ChatBot object
    # bot = HuggingFaceChatBotChroma(llm_config=REDPAJAMA_3B, disable_mem=True)
    # bot = HuggingFaceChatBotChroma(llm_config=REDPAJAMA_7B, disable_mem=True)
    # bot = HuggingFaceChatBotChroma(llm_config=VICUNA_7B, disable_mem=True)

    bot = HuggingFaceChatBotChroma(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=False)
    # bot = HuggingFaceChatBotChroma(llm_config=OPENORCA_MISTRAL_7B_Q5, disable_mem=True, gpu=True, gpu_layers=10)

    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_7B)
    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_16K_7B)
    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_LONGCHAT_1_5_32K_7B)

    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_7B, disable_mem=True)
    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_16K_7B, disable_mem=True)
    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_LONGCHAT_1_5_32K_7B, disable_mem=True)

    # GGUF Quantantized LLM, use less RAM
    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB
    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_16K_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB

    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_13B_Q6, disable_mem=True, gpu_layers=10) # mem = 18GB
    # bot = HuggingFaceChatBotChroma(llm_config=LMSYS_VICUNA_1_5_16K_13B_Q6, disable_mem=True, gpu_layers=0) # mem = 18GB

    # bot = HuggingFaceChatBotChroma(llm_config=WIZARDCODER_3B, disable_mem=True)
    # bot = HuggingFaceChatBotChroma(llm_config=WIZARDCODER_15B_Q8, disable_mem=True, gpu_layers=10) # mem = 23GB
    # bot = HuggingFaceChatBotChroma(llm_config=WIZARDCODER_PY_7B, disable_mem=True, gpu_layers=10)
    # bot = HuggingFaceChatBotChroma(llm_config=WIZARDCODER_PY_7B_Q6, disable_mem=True, gpu_layers=10) # mem = 9GB
    # bot = HuggingFaceChatBotChroma(llm_config=WIZARDCODER_PY_13B_Q6, disable_mem=True, gpu_layers=10) # mem = 14GB
    # bot = HuggingFaceChatBotChroma(llm_config=WIZARDCODER_PY_34B_Q5, disable_mem=True, gpu_layers=10) # mem = 27GB
    
    # This one is not good at all
    # bot = HuggingFaceChatBotChroma(llm_config=WIZARDLM_FALCON_40B_Q6K, disable_mem=True, gpu_layers=10) # mem = 45GB
    
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
