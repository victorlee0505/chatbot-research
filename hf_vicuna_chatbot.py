import logging
import os
import sys
import time
from typing import Dict, Union, Any, List

import numpy as np
import torch
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains.chat_vector_db.prompts import QA_PROMPT
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, pipeline)

from constants import CHROMA_SETTINGS_HF, PERSIST_DIRECTORY_HF
from ingest import Ingestion

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

persist_directory = PERSIST_DIRECTORY_HF
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
class VicunaChatBot:
    # initialize
    def __init__(
        self,
        model: str = None,
        load_data: bool = False,
        show_stream: bool = False,
        show_source: bool = False,
        show_callback: bool = False,
        gpu: bool = False,
        gui_mode: bool = False,
    ):
        self.model = model
        self.load_data = load_data
        self.show_stream = show_stream
        self.show_source = show_source
        self.show_callback = show_callback
        self.gpu = gpu
        self.gui_mode = gui_mode
        self.llm = None
        self.retriever = None
        self.embedding_llm = None
        self.qa = None
        self.ai_prefix = None
        self.human_prefix = None
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
        self.ingest_documents()
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

    def ingest_documents(self):
        offline = True
        if self.load_data:
            Ingestion(offline=offline)
        else:
            if os.path.exists(persist_directory):
                if os.listdir(persist_directory):
                    logger.info(f"Ingestion skipped!")
                else:
                    logger.info("PERSIST_DIRECTORY is empty.")
                    Ingestion(offline=offline)
            else:
                logger.info("PERSIST_DIRECTORY does not exist.")
                Ingestion(offline=offline)

    def initialize_model(self):
        logger.info("Initializing Model ...")
        self.embedding_llm = HuggingFaceEmbeddings()
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_llm,
            client_settings=CHROMA_SETTINGS_HF,
        )
        self.retriever = db.as_retriever(
            search_type="similarity", search_kwargs={"k": target_source_chunks}, max_tokens_limit=1000
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model, model_max_length=2048)
        if self.gpu:
            model = AutoModelForCausalLM.from_pretrained(self.model)
            model.half().cuda()
            torch_dtype = torch.float16
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model)
            torch_dtype = torch.bfloat16

        if self.gpu:
            stop_words_ids = [
                tokenizer(stop_word, return_tensors="pt").to("cuda")["input_ids"].squeeze()
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

        # self.qa = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     return_source_documents=self.show_source,
        # )

        # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""<human>: {question}\n<bot>:""")

        # question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        # doc_chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=QA_PROMPT)
        instruct_prefix = "instruct"
        if instruct_prefix.lower() in self.model.lower():
            self.ai_prefix="Q: "
            self.human_prefix="A: "
        else:
            self.ai_prefix="<bot>: "
            self.human_prefix="<human>: "
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            output_key="answer",
            memory_key="chat_history",
            ai_prefix=self.ai_prefix,
            human_prefix=self.human_prefix,
        )

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            get_chat_history=lambda h: h,
            return_source_documents=self.show_source,
        )

        # self.qa = ConversationalRetrievalChain(retriever=retriever, combine_docs_chain=doc_chain, question_generator=question_generator, return_source_documents= self.show_source)

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
        elif text.lower().strip() in ["reset"]:
            logger.info("<bot>: reset conversation memory detected.")
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=1000,
                output_key="answer",
                memory_key="chat_history",
                ai_prefix=self.ai_prefix,
                human_prefix=self.human_prefix,
            )
            self.qa.memory = memory
            self.inputs = text
        else:
            self.inputs = text

    def bot_response(self) -> str:
        if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.gui_mode:
            # a closing comment
            answer = "<bot>: See you soon! Bye!"
            print(f"<bot>: {answer}")
            return answer
        if self.inputs.lower().strip() in ["reset"]:
            # a closing comment
            answer = "<bot>: Conversation Memory cleared!"
            print(f"<bot>: {answer}")
            return answer
        response = self.qa({"question": self.inputs})
        answer, docs = (
            response["answer"],
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
            for document in docs:
                print(f"<bot>: source_documents")
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
        return answer

    # in case there is no response from model
    def random_response(self):
        return "I don't know", "I am not sure"


if __name__ == "__main__":
    # build a ChatBot object
    bot = VicunaChatBot()
    # bot = VicunaChatBot(show_source=True, show_callback=True)
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
