import gc

import streamlit as st
import torch
from streamlit_chat import message

from azure_chatbot import AzureOpenAiChatBot
from azure_chatbot_base import AzureOpenAiChatBotBase
from hf_chatbot_base import HuggingFaceChatBotBase
from hf_chatbot_chroma import HuggingFaceChatBotChroma
from hf_chatbot_coder import HuggingFaceChatBotCoder
from hf_llm_config import (
    CODEGEN25_7B,
    CODEGEN2_1B,
    CODEGEN2_4B,
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
    SANTA_CODER_1B,
    STARCHAT_BETA_16B_Q5,
    WIZARDCODER_3B,
    WIZARDCODER_15B_Q8,
    WIZARDCODER_PY_7B,
    WIZARDCODER_PY_7B_Q6,
    WIZARDCODER_PY_13B_Q6,
    WIZARDCODER_PY_34B_Q5,
    WIZARDLM_FALCON_40B_Q6K
)
from openai_chatbot import OpenAiChatBot
from langchain.callbacks.base import BaseCallbackHandler
from app_persist import load_widget_state, persist
from app_ui_constants import CHAT_ONLY, CLOSED, OPEN, REDPAJAMA_CHAT_3B_CONSTANT, SANTA_CODER_1B_CONSTANT
from hf_streaming_util import text_streamer

st.set_page_config(
        page_title="ChatBot-research",
        page_icon="👋",
    )

llm_chat_options = {
    "RedPajama 3B": REDPAJAMA_3B,
    "RedPajama 7B": REDPAJAMA_7B,
    "Vicuna 7B": VICUNA_7B,
    "Vicuna 1.5 7B LLAMA2": LMSYS_VICUNA_1_5_7B,
    "Vicuna 1.5 7B 16K LLAMA2": LMSYS_VICUNA_1_5_16K_7B,
    "Vicuna 1.5 7B 32K LLAMA2": LMSYS_LONGCHAT_1_5_32K_7B,
    "Vicuna 1.5 7B LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_7B_Q8,
    "Vicuna 1.5 7B 16K LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_16K_7B_Q8,
    "Vicuna 1.5 13B LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_13B_Q6,
    "Vicuna 1.5 13B 16K LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_16K_13B_Q6,
    "WizardCoder 3B": WIZARDCODER_3B,
    "WizardCoder 15B Q8": WIZARDCODER_15B_Q8,
    "WizardCoder Python 7B": WIZARDCODER_PY_7B,
    "WizardCoder Python 7B Q6": WIZARDCODER_PY_7B_Q6,
    "WizardCoder Python 13B Q6": WIZARDCODER_PY_13B_Q6,
    "WizardCoder Python 34B Q5": WIZARDCODER_PY_34B_Q5,
    "StarChat Beta 16B Q5 GGUF": STARCHAT_BETA_16B_Q5,
    "WizardLM Falcon 40B Q6K GGUF": WIZARDLM_FALCON_40B_Q6K,
}

llm_chroma_options = {
    "RedPajama 3B": REDPAJAMA_3B,
    "Vicuna 7B": VICUNA_7B,
    "Vicuna 1.5 7B LLAMA2": LMSYS_VICUNA_1_5_7B,
    "Vicuna 1.5 7B 16K LLAMA2": LMSYS_VICUNA_1_5_16K_7B,
    "Vicuna 1.5 7B 32K LLAMA2": LMSYS_LONGCHAT_1_5_32K_7B,
    "Vicuna 1.5 7B LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_7B_Q8,
    "Vicuna 1.5 7B 16K LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_16K_7B_Q8,
    "Vicuna 1.5 13B LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_13B_Q6,
    "Vicuna 1.5 13B 16K LLAMA2 Q8 GGUF": LMSYS_VICUNA_1_5_16K_13B_Q6,
    "WizardLM Falcon 40B Q6K GGUF": WIZARDLM_FALCON_40B_Q6K,
}

llm_coder_options = {
    "SantaCoder 1B": SANTA_CODER_1B,
    "Codegen2 1B": CODEGEN2_1B,
    "Codegen2 4B": CODEGEN2_4B,
    "Codegen2.5 7B": CODEGEN25_7B,
    "WizardCoder 3B": WIZARDCODER_3B,
    "WizardCoder Python 7B": WIZARDCODER_PY_7B,
}

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def main():
    print("run main()")
    if "page" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "page": "home",

            # Radio, selectbox and multiselect options.
            "options": ["Hello", "Everyone", "Happy", "Streamlit-ing"],
            "chat_mode_azure_options": [CHAT_ONLY, CLOSED, OPEN],
            "chat_mode_openai_options": [CLOSED],
            "chat_mode_hf_options": [CHAT_ONLY, OPEN],
            # "chat_model_hf_options": [REDPAJAMA_3B, REDPAJAMA_7B, VICUNA_7B],

            # Default widget values.
            "text": "",
            "slider": 0,
            "checkbox": False,
            "radio": "Hello",
            "selectbox": "Hello",
            "multiselect": ["Hello", "Everyone"],

            "chat_bot_azure": None,
            "chat_mode_azure": CHAT_ONLY,
            "chat_start_azure": False,
            "chat_azure": [],

            "chat_bot_openai": None,
            "chat_mode_openai": CLOSED,
            "chat_start_openai": False,
            "chat_openai": [],

            "chat_bot_hf": None,

            "chat_mode_hf": CHAT_ONLY,
            "chat_start_hf": False,
            "chat_model_hf": REDPAJAMA_CHAT_3B_CONSTANT,
            "chat_gpu_hf": False,
            "chat_no_mem_hf": False,
            "chat_hf": [],

            "chat_mode_chroma_hf": OPEN,
            "chat_start_chroma_hf": False,
            "chat_model_chroma_hf": REDPAJAMA_CHAT_3B_CONSTANT,
            "chat_gpu_chroma_hf": False,
            "chat_no_mem_chroma_hf": False,
            "chat_chroma_hf": [],

            "chat_mode_coder_hf": CHAT_ONLY,
            "chat_start_coder_hf": False,
            "chat_model_coder_hf": SANTA_CODER_1B_CONSTANT,
            "chat_gpu_coder_hf": False,
            "chat_no_mem_coder_hf": False,
            "chat_coder_hf": [],
        })

    page = st.sidebar.radio("Select your page", tuple(PAGES.keys()), format_func=str.capitalize, key="sidebar")

    PAGES[page]()

def page_home():

    st.write("# Welcome to playground! 👋")

    st.markdown(
        """

        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a chatbot from the sidebar** to start
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

def page_settings():
    st.header("Current App Status")

    st.write(
        f"""
        Azure Chatbot Status values
        ---------------

        - **chat_mode_azure**: `{st.session_state.chat_mode_azure}`
        - **chat_start_azure**: `{st.session_state.chat_start_azure}`

        OpenAI Chatbot Status values
        ---------------

        - **chat_mode_openai**: `{st.session_state.chat_mode_openai}`
        - **chat_start_openai**: `{st.session_state.chat_start_openai}`
        
        HuggingFace Chatbot Status values
        ---------------

        - **chat_mode_hf**: `{st.session_state.chat_mode_hf}`
        - **chat_start_hf**: `{st.session_state.chat_start_hf}`
        - **chat_model_hf**: `{st.session_state.chat_model_hf}`
        - **chat_gpu_hf**: `{st.session_state.chat_gpu_hf}`

        HuggingFace Chatbot Chroma Status values
        ---------------

        - **chat_mode_hf**: `{st.session_state.chat_mode_chroma_hf}`
        - **chat_start_hf**: `{st.session_state.chat_start_chroma_hf}`
        - **chat_model_hf**: `{st.session_state.chat_model_chroma_hf}`
        - **chat_gpu_hf**: `{st.session_state.chat_gpu_chroma_hf}`

        Redpajama Chatbot Coder Status values
        ---------------

        - **chat_mode_hf**: `{st.session_state.chat_mode_coder_hf}`
        - **chat_start_hf**: `{st.session_state.chat_start_coder_hf}`
        - **chat_model_hf**: `{st.session_state.chat_model_coder_hf}`
        - **chat_gpu_hf**: `{st.session_state.chat_gpu_coder_hf}`
        """
    )

def page_azure():

    def callback_reset():
        print("callback_reset")
        st.session_state["chat_start_azure"] = False
        st.session_state["chat_mode_azure"] = CHAT_ONLY
        del st.session_state["chat_bot_azure"]
        st.session_state["chat_bot_azure"] = None
        st.session_state["chat_azure"] = []
        st.session_state["chat_azure"] = []
        gc.collect()

    st.button("Reset Chatbot", on_click=callback_reset)

    def callback():
        print("callback")
        if st.session_state["chat_mode_azure"] in st.session_state["chat_mode_azure_options"]:
            st.session_state["chat_start_azure"] = True

    def home():
        # st.sidebar.header("Azure Chatbot")

        st.markdown(
            """

        Azure OpenAI powered chatbot w/ w/o ChromaDB vector store.

        Please complete the setting below!

        ### Chat mode?
        - Chat Only: this is normal chatbot
        - Closed : this will load context (docs / code repo) from source_documents folder and only answer to prompt related to the context
        - Open-Ended : this will load context (docs / code repo) from source_documents folder and answer to any prompt 

        ### Prompt:
        - If you choose Open-Ended chat mode, the response might not from the dataset you loaded.
        - you should ask like "in the context, do you find 'PLACEHOLDER'?

        """
        )

        st.radio("Choose a chat mode:", st.session_state["chat_mode_azure_options"], key=persist("chat_mode_azure"))

        start = st.button("Start", on_click=callback)

    def run():

        chatbot = st.session_state["chat_bot_azure"]

        ################################################################

        for message in st.session_state["chat_azure"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How may I help you?"):
            st.session_state["chat_azure"].append({"role": "user", "content": prompt})
            chatbot.user_input(prompt=prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                stream_handler = StreamHandler(message_placeholder)
                full_response = chatbot.qa.run(prompt, callbacks=[stream_handler])
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state["chat_azure"].append({"role": "assistant", "content": full_response})

        #################################################################

    if "chat_start_azure" not in st.session_state:
        if st.session_state["chat_bot_azure"] is not None:
            st.session_state["chat_start_azure"] = True
        else:
            st.session_state["chat_start_azure"] = False
            
    if st.session_state["chat_start_azure"] == False:
        print("Welcome to Azure Chatbot")
        home()
    else:
        print("Starting Azure Chatbot")
        if st.session_state["chat_mode_azure"] is not None:
            # try to load model
            if (
                st.session_state["chat_mode_azure"] == CHAT_ONLY
                and st.session_state["chat_bot_azure"] is None
            ):
                st.session_state["chat_bot_azure"] = AzureOpenAiChatBotBase(gui_mode=True)
                print(f"Azure Chatbot: {CHAT_ONLY}")
            elif (st.session_state["chat_mode_azure"] == CLOSED
                and st.session_state["chat_bot_azure"] is None
            ):
                st.session_state["chat_bot_azure"] = AzureOpenAiChatBot(gui_mode=True)
                print(f"Azure Chatbot: {CLOSED}")
            elif (st.session_state["chat_mode_azure"] == OPEN
                and st.session_state["chat_bot_azure"] is None
            ):
                st.session_state["chat_bot_azure"] = AzureOpenAiChatBot(gui_mode=True, open_chat=True)
                print(f"Azure Chatbot: {OPEN}")
            else:
                if st.session_state["chat_bot_azure"] is None:
                    st.session_state["chat_bot_azure"] = AzureOpenAiChatBotBase(gui_mode=True)
                    print(f"Azure Chatbot: {CHAT_ONLY}")
            run()
        else:
            home()

def page_openai():

    def callback_reset():
        print("callback_reset")
        st.session_state["chat_start_openai"] = False
        st.session_state["chat_mode_openai"] = CLOSED
        del st.session_state["chat_bot_openai"]
        st.session_state["chat_bot_openai"] = None
        st.session_state["chat_openai"] = []
        gc.collect()

    st.button("Reset Chatbot", on_click=callback_reset)

    def callback():
        print("callback")
        if st.session_state["chat_mode_openai"] in st.session_state["chat_mode_openai_options"]:
            st.session_state["chat_start_openai"] = True

    def home():
        # st.sidebar.header("Azure Chatbot")

        st.markdown(
            """
        OpenAI powered chatbot with ChromaDB vector store.

        Please complete the setting below!

        ### Chat mode?  Only 1 mode is available, Base Chatbot just use ChatGPT

        - Closed : this will load context (docs / code repo) from source_documents folder and only answer to prompt related to the context

        """
        )

        st.radio("Choose a chat mode:", st.session_state["chat_mode_openai_options"], key=persist("chat_mode_openai"))

        start = st.button("Start", on_click=callback)

    def run():

        chatbot = st.session_state["chat_bot_openai"]

        ################################################################

        for message in st.session_state["chat_openai"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How may I help you?"):
            st.session_state["chat_openai"].append({"role": "user", "content": prompt})
            chatbot.user_input(prompt=prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = chatbot.bot_response()
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state["chat_openai"].append({"role": "assistant", "content": full_response})

        #################################################################

    if "chat_start_openai" not in st.session_state:
        if st.session_state["chat_bot_openai"] is not None:
            st.session_state["chat_start_openai"] = True
        else:
            st.session_state["chat_start_openai"] = False
            
    if st.session_state["chat_start_openai"] == False:
        print("Welcome to OpenAI Chatbot")
        home()
    else:
        print("Starting OpenAI Chatbot")
        if st.session_state["chat_mode_openai"] is not None:
            # try to load model

            if (st.session_state["chat_mode_openai"] == CLOSED
                and st.session_state["chat_bot_openai"] is None
            ):
                st.session_state["chat_bot_openai"] = OpenAiChatBot(gui_mode=True)
                print(f"Azure Chatbot: {CLOSED}")
            else:
                if st.session_state["chat_bot_openai"] is None:
                    st.session_state["chat_bot_openai"] = OpenAiChatBot(gui_mode=True)
                    print(f"Azure Chatbot: {CLOSED}")
            run()
        else:
            home()

def page_hf():

    def callback_reset():
        print("callback_reset")
        st.session_state["chat_start_hf"] = False
        st.session_state["chat_mode_hf"] = CHAT_ONLY
        st.session_state["chat_model_hf"] = REDPAJAMA_CHAT_3B_CONSTANT
        st.session_state["chat_gpu_hf"] = False
        st.session_state["chat_no_mem_hf"] = False
        del st.session_state["chat_bot_hf"]
        st.session_state["chat_bot_hf"] = None
        st.session_state["chat_hf"] = []
        gc.collect()
        torch_gc()

    st.button("Reset Chatbot", on_click=callback_reset)

    def callback():
        print("callback")
        if st.session_state["chat_mode_hf"] in st.session_state["chat_mode_hf_options"]:
            st.session_state["chat_start_hf"] = True

    def home():
        st.markdown(
            """

        HuggingFace Opensource LLM powered chatbot.

        Please complete the setting below!

        ##### Model?
        - 3B: a model that can be run with 16GB system memory or at least 8GB VRAM with CUDA
        - 7B: a model that can be run with 40GB system memory or at least 16GB VRAM with CUDA

        ##### CUDA (Nvidia GPU acceleration) (if you don't know what is this, leave it un-checked)?
        - Default: False
        - True: if you want to try to use CUDA

        ##### GGUF Quantization Model
        - Model name that has "GGUF" is a model that has been quantized to lower precision to reduce memory usage with quality trade-off
        - 7B Q8 model use aboout 10GB of RAM vs 7B model that use about 40GB of RAM
        - 13B Q8 model use aboout 20GB of RAM vs 13B model that use about 60GB of RAM
        - StarChat 16B Q8 model use aboout 23GB of RAM
        - WizardLM Falcon 40B Q6K model use aboout 45GB of RAM

        """
        )
        # Define the columns

        # Add a radio button to the first column

        st.selectbox("Choose a LLM model:", llm_chat_options.keys(), key=persist("chat_model_hf"))
        # Add a checkbox to the second column

        st.checkbox("CUDA", key=persist("chat_gpu_hf"))
        st.checkbox("Disable Chat Memory", key=persist("chat_no_mem_hf"))
        st.session_state["chat_mode_hf"] = CHAT_ONLY
        # st.radio("Choose a chat mode:", st.session_state["chat_mode_hf_options"], key=persist("chat_mode_hf"))
        start = st.button("Start", on_click=callback)
        if start:
            print("start clicked")

    def run():
        chatbot : HuggingFaceChatBotBase = st.session_state["chat_bot_hf"]

        ################################################################

        for message in st.session_state["chat_hf"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How may I help you?"):
            st.session_state["chat_hf"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in text_streamer(prompt, chatbot):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state["chat_hf"].append({"role": "assistant", "content": full_response})

        #################################################################

    if "chat_start_hf" not in st.session_state:
        if st.session_state["chat_bot_hf"] is not None:
            st.session_state["chat_start_hf"] = True
        else:
            st.session_state["chat_start_hf"] = False
            
    if st.session_state["chat_start_hf"] == False:
        print("Welcome to Huggingface Chatbot")
        home()
    else:
        print("Starting HuggingFace Chatbot")
        if st.session_state["chat_mode_hf"] is not None:
            print(st.session_state["chat_mode_hf"])
            # try to load model
            if st.session_state["chat_bot_hf"] is None:
                gpu_layers= 10 if st.session_state["chat_gpu_hf"] else 0
                st.session_state["chat_bot_hf"] = HuggingFaceChatBotBase(llm_config=llm_chat_options.get(st.session_state["chat_model_hf"]), gpu= st.session_state["chat_gpu_hf"], gpu_layers=gpu_layers, server_mode=True, disable_mem=st.session_state["chat_no_mem_hf"])
                name = st.session_state["chat_model_hf"]
                print(f"{name} Chatbot: {CHAT_ONLY}")
            run()
        else:
            home()

def page_hf_chroma():

    def callback_reset():
        print("callback_reset")
        st.session_state["chat_start_chroma_hf"] = False
        st.session_state["chat_mode_chroma_hf"] = OPEN
        st.session_state["chat_model_chroma_hf"] = REDPAJAMA_CHAT_3B_CONSTANT
        st.session_state["chat_gpu_chroma_hf"] = False
        del st.session_state["chat_bot_hf"]
        st.session_state["chat_bot_hf"] = None
        st.session_state["chat_chroma_hf"] = []
        gc.collect()
        torch_gc()

    st.button("Reset Chatbot", on_click=callback_reset)

    def callback():
        print("callback")
        if st.session_state["chat_mode_chroma_hf"] in st.session_state["chat_mode_hf_options"]:
            st.session_state["chat_start_chroma_hf"] = True

    def home():
        st.markdown(
            """

        HuggingFace Opensource LLM powered chatbot with ChromaDB vector store.

        Please complete the setting below!

        ##### Model?
        - 3B: a model that can be run with 16GB system memory or at least 8GB VRAM with CUDA
        - 7B: a model that can be run with 40GB system memory or at least 16GB VRAM with CUDA

        ##### CUDA (Nvidia GPU acceleration) (if you don't know what is this, leave it un-checked)?
        - Default: False
        - True: if you want to try to use CUDA

        ##### GGUF Quantization Model
        - Model name that has "GGUF" is a model that has been quantized to lower precision to reduce memory usage
        - 7B Q8 model use aboout 10GB of RAM vs 7B model that use about 40GB of RAM
        - 13B Q8 model use aboout 20GB of RAM vs 13B model that use about 60GB of RAM
        - WizardLM Falcon 40B Q6K model use aboout 45GB of RAM

        """
        )
        # Define the columns

        # Add a radio button to the first column

        st.selectbox("Choose a LLM model:", llm_chroma_options.keys(), key=persist("chat_model_chroma_hf"))
        # Add a checkbox to the second column

        st.checkbox("CUDA", key=persist("chat_gpu_chroma_hf"))
        st.checkbox("Disable Chat Memory", key=persist("chat_no_mem_chroma_hf"))
        st.session_state["chat_bot_chroma_hf"] = OPEN
        # st.radio("Choose a chat mode:", st.session_state["chat_mode_hf_options"], key=persist("chat_mode_hf"))
        start = st.button("Start", on_click=callback)
        if start:
            print("start clicked")

    def run():
        chatbot : HuggingFaceChatBotChroma = st.session_state["chat_bot_hf"]

        ################################################################

        for message in st.session_state["chat_chroma_hf"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How may I help you?"):
            st.session_state["chat_chroma_hf"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in text_streamer(prompt, chatbot):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state["chat_chroma_hf"].append({"role": "assistant", "content": full_response})

        #################################################################

    if "chat_start_chroma_hf" not in st.session_state:
        if st.session_state["chat_bot_hf"] is not None:
            st.session_state["chat_start_chroma_hf"] = True
        else:
            st.session_state["chat_start_chroma_hf"] = False
            
    if st.session_state["chat_start_chroma_hf"] == False:
        print("Welcome to Huggingface Chatbot Chroma")
        home()
    else:
        print("Starting HuggingFace Chatbot")
        if st.session_state["chat_mode_chroma_hf"] is not None:
            print(st.session_state["chat_mode_hf"])
            # try to load model
            if st.session_state["chat_bot_hf"] is None:
                gpu_layers= 10 if st.session_state["chat_gpu_chroma_hf"] else 0
                st.session_state["chat_bot_hf"] = HuggingFaceChatBotChroma(llm_config=llm_chroma_options.get(st.session_state["chat_model_chroma_hf"]), gpu_layers=gpu_layers, gpu= st.session_state["chat_gpu_chroma_hf"], server_mode=True, disable_mem=st.session_state["chat_no_mem_hf"])
                name = st.session_state["chat_model_chroma_hf"]
                print(f"{name} Chatbot: {OPEN}")
            run()
        else:
            home()

def page_hf_coder():

    def callback_reset():
        print("callback_reset")
        st.session_state["chat_start_coder_hf"] = False
        st.session_state["chat_mode_coder_hf"] = CHAT_ONLY
        st.session_state["chat_model_coder_hf"] = SANTA_CODER_1B_CONSTANT
        st.session_state["chat_gpu_coder_hf"] = False
        del st.session_state["chat_bot_hf"]
        st.session_state["chat_bot_hf"] = None
        st.session_state["chat_coder_hf"] = []
        gc.collect()
        torch_gc()

    st.button("Reset Chatbot", on_click=callback_reset)

    def callback():
        print("callback")
        if st.session_state["chat_mode_coder_hf"] in st.session_state["chat_mode_hf_options"]:
            st.session_state["chat_start_coder_hf"] = True

    def home():
        st.markdown(
            """

        HuggingFace Opensource LLM powered chatbot for Coding.
            
        Please complete the setting below!

        ##### Model?
        - 3B: a model that can be run with 16GB system memory or at least 8GB VRAM with CUDA
        - 7B: a model that can be run with 40GB system memory or at least 16GB VRAM with CUDA

        ##### CUDA (Nvidia GPU acceleration) (if you don't know what is this, leave it un-checked)?
        - Default: False
        - True: if you want to try to use CUDA

        ##### Chat mode?
        - Chat Only: this is normal chatbot
        - Open-Ended : this will load context (docs / code repo) from source_documents folder and answer to any prompt 

        ##### Prompt:
        - If you choose Open-Ended chat mode, the response might not from the dataset you loaded.
        - you should ask like "in the context, do you find 'PLACEHOLDER'?

        """
        )
        # Define the columns

        # Add a radio button to the first column

        st.selectbox("Choose a LLM model:", llm_coder_options.keys(), key=persist("chat_model_coder_hf"))
        # Add a checkbox to the second column

        st.checkbox("CUDA", key=persist("chat_gpu_coder_hf"))
        st.session_state["chat_mode_coder_hf"] = CHAT_ONLY
        # st.radio("Choose a chat mode:", st.session_state["chat_mode_hf_options"], key=persist("chat_mode_hf"))
        start = st.button("Start", on_click=callback)
        if start:
            print("start clicked")

    def run():
        chatbot : HuggingFaceChatBotCoder = st.session_state["chat_bot_hf"]

        ################################################################

        for message in st.session_state["chat_coder_hf"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("How may I help you?"):
            st.session_state["chat_coder_hf"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in text_streamer(prompt, chatbot):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state["chat_coder_hf"].append({"role": "assistant", "content": full_response})

        #################################################################

    if "chat_start_coder_hf" not in st.session_state:
        if st.session_state["chat_bot_hf"] is not None:
            st.session_state["chat_start_coder_hf"] = True
        else:
            st.session_state["chat_start_coder_hf"] = False
            
    if st.session_state["chat_start_coder_hf"] == False:
        print("Welcome to Huggingface Chatbot Coder")
        home()
    else:
        print("Starting HuggingFace Chatbot")
        if st.session_state["chat_mode_coder_hf"] is not None:
            print(st.session_state["chat_mode_coder_hf"])
            # try to load model
            if st.session_state["chat_bot_hf"] is None:
                st.session_state["chat_bot_hf"] = HuggingFaceChatBotCoder(llm_config=llm_coder_options.get(st.session_state["chat_model_coder_hf"]), gpu= st.session_state["chat_gpu_coder_hf"], server_mode=True)
                name = st.session_state["chat_model_coder_hf"]
                print(f"{name} Chatbot: {CHAT_ONLY}")
            run()
        else:
            home()

PAGES = {
    "home": page_home,
    "Status": page_settings,
    "azure-chatbot": page_azure,
    "openai-chatbot": page_openai,
    "huggingface-chatbot": page_hf,
    "huggingface-chroma-chatbot": page_hf_chroma,
    "huggingface-coder-chatbot": page_hf_coder,
}

if __name__ == "__main__":
    load_widget_state()
    main()
