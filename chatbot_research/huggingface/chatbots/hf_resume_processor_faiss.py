import json
import logging
import os
import random
import shutil
import time
from typing import Union

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from chatbot_research.huggingface.config.hf_llm_config import LLMConfig
from chatbot_research.huggingface.inference.hf_chat_llama_cpp import HFChatllamaCpp
from chatbot_research.ingestion.ingest_constants import (
    ALL_MINILM_L6_V2,
    CHROMA_SETTINGS_AZURE,
    CHROMA_SETTINGS_HF,
    PERSIST_DIRECTORY_AZURE,
    PERSIST_DIRECTORY_HF,
    STELLA_EN_1_5B_V5,
)
from chatbot_research.ingestion.ingest_faiss import IngestionFAISS


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # start time before function executes
        result = func(*args, **kwargs)  # execute function
        end_time = time.time()  # end time after function executes
        exec_time = end_time - start_time  # execution time
        args[0].logger.info(f"Executed {func.__name__} in {exec_time:.4f} seconds")
        return result

    return wrapper


persist_directory = PERSIST_DIRECTORY_HF


# A ChatBot class
# Build a ChatBot class with all necessary modules to make a complete conversation
class HuggingFaceChatbotResumeProcessorFaiss:
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
        self.rag_chain = None
        self.embedding_llm = None

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

    def create_vectorstore(self, path) -> FAISS | None:
        ingest = IngestionFAISS(
            offline=True,
            embedding_model=STELLA_EN_1_5B_V5,
            persist_directory=path,
        )
        return ingest.load_vectorstore()

    def create_model(self):
        if self.llm_config:
            self.llm_config.validate()
        self.logger.info("Initializing ChatBot ...")
        if self.gpu:
            model_kwargs = {"trust_remote_code": True}
            self.embedding_llm = HuggingFaceEmbeddings(
                model_kwargs=model_kwargs, model_name=STELLA_EN_1_5B_V5
            )
        else:
            model_kwargs = {"trust_remote_code": True}
            self.embedding_llm = HuggingFaceEmbeddings(
                model_kwargs=model_kwargs, model_name=STELLA_EN_1_5B_V5
            )

        self.inference = HFChatllamaCpp(
            logger=self.logger,
            llm_config=self.llm_config,
            gpu=self.gpu,
            server_mode=self.server_mode,
            disable_mem=self.disable_mem,
            show_callback=self.show_callback,
        )

        self.llm = self.inference.initialize_chat_llm(temperature=0.5)

    @timer_decorator
    def process_resume(self, vectorstore: VectorStore):

        descriptions = {
            key: value["description"]
            for key, value in self.dict_schema["function"]["parameters"][
                "properties"
            ].items()
        }

        self.logger.info(f"descriptions: {descriptions}")

        text = self.ask_for_prompt(descriptions=descriptions, ask_for=None)

        self.logger.info(f"ask_for: {text}")

        # EXTRACTION_PROMPT_TEMPLATE = """
        #     <|im_start|>system
        #     You are expert on extract information into JSON. Here's the json schema you must adhere to:\n<schema>\n{{schema}}\n</schema> \
        #     <|im_end|>
        #     <|im_start|>user
        #     Extracts the relevant information from the passage. If you do not know the value of an attribute then return empty string for the attribute's value. \
        #     ### Passage: {context}
        #     Question: {question}
        #     <|im_end|>
        #     <|im_start|>assistant
        #     """

        # EXTRACTION_PROMPT_TEMPLATE = EXTRACTION_PROMPT_TEMPLATE.replace(
        #     "{schema}", json.dumps(self.pydantic_model_object.model_dump(mode='json'))
        #     )

        # self.logger.info(f"EXTRACTION_PROMPT_TEMPLATE: {EXTRACTION_PROMPT_TEMPLATE}")

        # EXTRACTION_PROMPT = ChatPromptTemplate.from_template(template=EXTRACTION_PROMPT_TEMPLATE)

        EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
            """
            <|im_start|>system
            You are expert on extract information.
            <|im_end|>
            <|im_start|>user
            Extracts the relevant information from the passage. If you do not know the value of an attribute then return empty string for the attribute's value. \
            ### Passage: {context}
            Question: {question}
            <|im_end|>
            <|im_start|>assistant 
            """
        )

        self.llm = self.inference.initialize_chat_llm(temperature=0)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.llm_config.target_source_chunks},
            max_tokens_limit=self.llm_config.retriever_max_tokens_limit,
        )

        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | EXTRACTION_PROMPT
            | self.llm.with_structured_output(self.dict_schema)
        )

        pydantic_object, ask_for = self.filter_response(
            text_input=text, pydantic_object=self.pydantic_model_object
        )

        self.pydantic_model_object = pydantic_object

        retry = 0
        while len(ask_for) > 0:
            pydantic_object, ask_for = self.filter_response(
                text_input=text, pydantic_object=self.pydantic_model_object
            )
            text = self.ask_for_prompt(descriptions=descriptions, ask_for=ask_for)
            self.logger.info(f"ask_for_prompt: {text}")
            if self.pydantic_model_object == pydantic_object:
                retry += 1
                if retry > 3:
                    break
            self.pydantic_model_object = pydantic_object
            self.logger.info(f"latest pydantic object: {self.pydantic_model_object}")

        self.logger.info("done processing resume")
        self.logger.info(f"finaly pydantic object: {self.pydantic_model_object}")
        return self.pydantic_model_object

    def ask_for_prompt(self, descriptions: dict = None, ask_for: list = None):

        if ask_for is not None:
            # pick description from descriptions using ask_for keys
            prompt = [descriptions[key] for key in ask_for]
            k_size = 3 if len(prompt) > 3 else len(prompt)
            key_string = random.sample(prompt, k=k_size)
            key_string = "\n".join(key_string)
        else:
            # all descriptions
            prompt = list(descriptions.values())
            k_size = 3 if len(prompt) > 3 else len(prompt)
            key_string = random.sample(prompt, k=k_size)
            key_string = "\n".join(key_string)

        result = f"{key_string}"
        return result

    def filter_response(self, text_input, pydantic_object: BaseModel):

        result = self.rag_chain.invoke(text_input)
        self.logger.info(f"rag chain result : {result}")
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

        non_empty_details = {
            k: v for k, v in new_details_dict.items() if v not in [None, ""]
        }
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

    def get_subfolders(self, path):
        # Get all entries in the directory
        entries = os.listdir(path)

        # Filter out only the subfolders
        return [
            os.path.join(path, entry)
            for entry in entries
            if os.path.isdir(os.path.join(path, entry))
        ]

    def ingest_resumes(self, staging_path: list = []):
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
                subfolder_path = [
                    os.path.join(staging_path, entry)
                    for entry in entries
                    if os.path.isdir(os.path.join(staging_path, entry))
                ]
                count = 0
                storages_path = []
                ingest: IngestionFAISS = None
                for path in subfolder_path:
                    persist_dir = persist_directory + f"/{count}"
                    storages_path.append(persist_dir)

                    if ingest is None:
                        ingest = IngestionFAISS(
                            offline=offline,
                            embedding_model=STELLA_EN_1_5B_V5,
                            persist_directory=persist_dir,
                        )
                        ingest.run_ingest()
                    else:
                        ingest.source_path = path
                        ingest.persist_directory = persist_dir
                        ingest.run_ingest()
                    count += 1
                return storages_path
