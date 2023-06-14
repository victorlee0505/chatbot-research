import logging
import os
import sys
import time

import numpy as np
import torch
from langchain import HuggingFaceHub, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

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
class RedpajamaChatBot:
    # initialize
    def __init__(
        self,
        model: str = None,
        load_data: bool = False,
        show_stream: bool = False,
        show_source: bool = False,
        gpu: bool = False,
        gui_mode: bool = False,
    ):
        self.model = model
        self.load_data = load_data
        self.show_stream = show_stream
        self.show_source = show_source
        self.gpu = gpu
        self.gui_mode = gui_mode
        self.llm = None
        self.llm_predictor = None
        self.embedding_llm = None
        self.prompt_helper = None
        self.service_context = None
        self.qa = None
        self.index = None
        self.query_engine = None
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
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

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

        # self.qa = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     return_source_documents=self.show_source,
        # )

        # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""<human>: {question}\n<bot>:""")

        # question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        # doc_chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=QA_PROMPT)

        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
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
        else:
            self.inputs = text

    def bot_response(self)-> str:
        if self.inputs.lower().strip() in ["bye", "quit", "exit"] and self.gui_mode:
            # a closing comment
            answer = "<bot>: See you soon! Bye!"
            print(f"<bot>: {answer}")
            return answer
        response = self.qa({"question": self.inputs, "chat_history": self.chat_history})
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
        self.chat_history[:] =  self.chat_history[-10:]
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
    bot = RedpajamaChatBot()
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
