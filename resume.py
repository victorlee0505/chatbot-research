import csv
import os
from pydantic import BaseModel, Field
import logging
import json
import os
import sys
import time
import shutil
from chromadb.config import Settings

import numpy as np
import openai
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.schema.vectorstore import VectorStoreRetriever

from chatbot_research.ingestion.ingest_constants import CHROMA_SETTINGS_AZURE, PERSIST_DIRECTORY_AZURE
from chatbot_research.ingestion.ingest import Ingestion

import logging
import os
import sys
import time
from typing import Any, Dict, List, Union

import chromadb
import ctransformers
import numpy as np
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
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
from chatbot_research.huggingface.config.hf_prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT_DOCUMENT_CHAT
from chatbot_research.ingestion.ingest import Ingestion
from chatbot_research.ingestion.ingest_constants import CHROMA_SETTINGS_HF, PERSIST_DIRECTORY_HF

persist_directory = PERSIST_DIRECTORY_HF + "_resume"

from student_details import StudentDetails

# Example data for a single student
student_data = {
    "student_id": "001",
    "first_name": "John",
    "Last Name": "Doe",
    "Email": "johndoe@example.com",
    "Phone Number": "555-555-5555",
    "Address": "123 Main St, City, State",
    "Graduation Year": "2023",
    "Major/Field of Study": "Computer Science",
    "University": "University of Example",
    "GPA": "3.8",
    "Skills": "Python, Java, SQL",
    "Experience": "Software Developer Intern",
    "Education": "Bachelor's in Computer Science",
    "Projects": "Web Application, Data Analysis",
    "Summary": "Detail-oriented computer science student with a passion for programming.",
}

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


class HuggingFaceChromaResumeProcessor:
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
        self.vectors = []
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
        # self.welcome()

    def start_model(self):
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
        # self.copy_pdfs_to_staging(src_folder="./resume_input", staging_folder="./staging")
        # self.ingest_resumes(staging_path="./staging")
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

    def get_subfolders(self, path):
        # Get all entries in the directory
        entries = os.listdir(path)
        
        # Filter out only the subfolders
        return [os.path.join(path, entry) for entry in entries if os.path.isdir(os.path.join(path, entry))]


    def copy_pdfs_to_staging(self, src_folder, staging_folder):
        # Ensure the staging folder exists
        if not os.path.exists(staging_folder):
            os.makedirs(staging_folder)

            # Iterate over all files in the source directory
            for filename in os.listdir(src_folder):
                # Create a new folder for this PDF in the staging directory
                new_folder = os.path.join(staging_folder, os.path.splitext(filename)[0])
                os.makedirs(new_folder, exist_ok=True)
                
                # Copy the PDF into the new folder
                shutil.copy2(os.path.join(src_folder, filename), new_folder)
        else:
            print("Staging folder already exists!")

    def ingest_resumes(self, staging_path : [] = None):
        if os.path.exists(persist_directory):
            self.logger.info(f"Ingestion skipped!")
            return self.get_subfolders(persist_directory)

        else:
            print("Ingesting started ...")
            if staging_path is not None:
                offline = True
                staging_path = "./staging"
                # Initialize an empty list to store subfolder paths
                entries = os.listdir(staging_path)
                subfolder_path = [os.path.join(staging_path, entry) for entry in entries if os.path.isdir(os.path.join(staging_path, entry))]
                count = 0
                storages_path = []
                for path in subfolder_path:
                    persist_dir = persist_directory+f"/{count}"
                    storages_path.append(persist_dir)
                    settings = Settings(
                        # chroma_db_impl='duckdb+parquet',
                        persist_directory=persist_dir,
                        anonymized_telemetry=False
)
                    Ingestion(offline=offline, chroma_setting=settings ,source_path=path, persist_directory=persist_dir)
                    count += 1
                return storages_path

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

    def initialize_gguf_model(self):
        self.logger.info("Initializing Model ...")

        if self.gpu:
            self.embedding_llm = HuggingFaceEmbeddings()
        else:
            model_kwargs = {'device': 'cpu'}
            self.embedding_llm = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

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

    def create_retriever(self, path):
        client = chromadb.PersistentClient(settings=CHROMA_SETTINGS_HF, path=path)
        db = Chroma(
            client=client,
            embedding_function=self.embedding_llm,
        )
        retriever = db.as_retriever(
            search_type="similarity", search_kwargs={"k": 20}, max_tokens_limit=self.llm_config.retriever_max_tokens_limit
        )
        return retriever

    def _create_qa_from_retriver(self, retriever):
        return RetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            prompt=self.llm_config.prompt_qa_template,
            return_source_documents=self.show_source,
        )

    def add_non_empty_details(self, current_details: StudentDetails, new_details: StudentDetails):
        non_empty_details = {k: v for k, v in new_details.model_dump().items() if v not in [None, ""]}
        updated_details = current_details.model_copy(update=non_empty_details)
        return updated_details

    def filter_response(self, text_input, user_details, llm):
        chain = create_tagging_chain_pydantic(pydantic_schema=StudentDetails, llm=llm)
        res = chain.run(text_input)
        # add filtered info to the
        user_details = self.add_non_empty_details(user_details,res)
        return user_details

    @timer_decorator
    def process_resume(self, retriever : VectorStoreRetriever):
        # receive input from user
        text = """
            Please populate the following columns for a student's resume:
            1. student_id:
            2. first_name:
            3. last_name:
            4. Email:
            5. phone_number:
            6. address:
            7. graduation_year:
            8. major:
            9. university:
            10. gpa:
            11. skills:
            12. experience:
            13. education:
            14. projects:
            15. summary:
        """
        self.inputs = text
        self.qa = self._create_qa_from_retriver(retriever)
        student = StudentDetails()
        if issubclass(StudentDetails, BaseModel):
            new_user_details = self.filter_response(text_input=self.inputs, user_details=student, llm=self.qa)
            return new_user_details
        else:
            print("Error: StudentDetails is not a subclass of BaseModel")




if __name__ == "__main__":
    # get config
    # build a ChatBot object
    # bot = HuggingFaceChatBotChroma(llm_config=REDPAJAMA_3B, disable_mem=True)
    # bot = HuggingFaceChatBotChroma(llm_config=REDPAJAMA_7B, disable_mem=True)
    # bot = HuggingFaceChatBotChroma(llm_config=VICUNA_7B, disable_mem=True)

    bot = HuggingFaceChromaResumeProcessor(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)
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
    
    students_detail = []

    folder_path = "./resume_input"
    staging_path = "./staging"

    bot.copy_pdfs_to_staging(src_folder=folder_path, staging_folder=staging_path)
    storages_path = bot.ingest_resumes(staging_path=staging_path)
    bot.start_model()
    for storage_path in storages_path:
        retriever = bot.create_retriever(path=storage_path)
        student_detail = bot.process_resume(retriever=retriever)
        students_detail.append(student_detail)


    directory_path = "resume_output"

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)

    print(f"Directory '{directory_path}' created successfully.")

    csv_filename = f"{directory_path}/student_data.csv"

    headings = [
        "student_id", "first_name", "last_name", "email", "phone_number",
        "address", "graduation_year", "major", "university",
        "gpa", "skills", "experience", "education", "projects", "summary"
    ]

    # Create and open the CSV file in write mode
    with open(csv_filename, mode='w', newline='') as csv_file:
        # Create a CSV writer object
        writer = csv.DictWriter(csv_file, fieldnames=headings)
        
        # Write the headings to the CSV file
        writer.writeheader()

        for student_data in students_detail:
            # Write the student data to the CSV file
            writer.writerow(student_data)

    print(f"CSV file '{csv_filename}' created successfully.")
