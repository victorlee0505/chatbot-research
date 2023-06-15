import gc
import os
import openai
import streamlit as st
from streamlit_chat import message

from azure_chatbot import AzureOpenAiChatBot
from azure_chatbot_base import AzureOpenAiChatBotBase
from openai_chatbot import OpenAiChatBot
from hf_redpajama_chatbot import RedpajamaChatBot
from hf_redpajama_chatbot_base import RedpajamaChatBotBase
from persist import load_widget_state, persist
from ui_constants import CHAT_ONLY, CLOSED, KILL_MESSAGE, OPEN, REDPAJAMA_CHAT_3B, REDPAJAMA_CHAT_7B

st.set_page_config(
        page_title="ChatBot-research",
        page_icon="👋",
    )

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
            "chat_mode_redpajama_options": [CHAT_ONLY, OPEN],
            "chat_model_redpajama_options": [REDPAJAMA_CHAT_3B, REDPAJAMA_CHAT_7B],

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
            "prompt_azure": [],
            "completion_azure": [],

            "chat_bot_openai": None,
            "chat_mode_openai": CLOSED,
            "chat_start_openai": False,
            "prompt_openai": [],
            "completion_openai": [],

            "chat_bot_redpajama": None,
            "chat_mode_redpajama": CHAT_ONLY,
            "chat_start_redpajama": False,
            "chat_model_redpajama": REDPAJAMA_CHAT_3B,
            "chat_gpu_redpajama": False,
            "prompt_redpajama": [],
            "completion_redpajama": [],
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
        
        Redpajama Chatbot Status values
        ---------------

        - **chat_mode_redpajama**: `{st.session_state.chat_mode_redpajama}`
        - **chat_start_redpajama**: `{st.session_state.chat_start_redpajama}`
        - **chat_model_redpajama**: `{st.session_state.chat_model_redpajama}`
        - **chat_gpu_redpajama**: `{st.session_state.chat_gpu_redpajama}`
        """
    )

def page_azure():

    def callback_reset():
        print("callback_reset")
        st.session_state["chat_start_azure"] = False
        st.session_state["chat_mode_azure"] = CHAT_ONLY
        del st.session_state["chat_bot_azure"]
        st.session_state["chat_bot_azure"] = None
        st.session_state["prompt_azure"] = []
        st.session_state["completion_azure"] = []
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
        # container for chat history
        response_container = st.container()
        # container for text box
        container = st.container()

        chatbot = st.session_state["chat_bot_azure"]

        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area("You:", key="input", height=100)
                submit_button = st.form_submit_button(label="Send")

            if submit_button and user_input:
                chatbot.user_input(prompt=user_input)
                output = chatbot.bot_response()
                print(f"output: {output}")
                st.session_state["prompt_azure"].append(user_input)
                st.session_state["completion_azure"].append(output)

        if st.session_state["completion_azure"]:
            with response_container:
                for i in range(len(st.session_state["completion_azure"])):
                    message(
                        st.session_state["prompt_azure"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )
                    message(st.session_state["completion_azure"][i], key=str(i))

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
        st.session_state["prompt_openai"] = []
        st.session_state["completion_openai"] = []
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

        Please complete the setting below!

        ### Chat mode?  Only 1 mode is available, Base Chatbot just use ChatGPT

        - Closed : this will load context (docs / code repo) from source_documents folder and only answer to prompt related to the context

        """
        )

        st.radio("Choose a chat mode:", st.session_state["chat_mode_openai_options"], key=persist("chat_mode_openai"))

        start = st.button("Start", on_click=callback)

    def run():
        # container for chat history
        response_container = st.container()
        # container for text box
        container = st.container()

        chatbot = st.session_state["chat_bot_openai"]

        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area("You:", key="input", height=100)
                submit_button = st.form_submit_button(label="Send")

            if submit_button and user_input:
                chatbot.user_input(prompt=user_input)
                output = chatbot.bot_response()
                print(f"output: {output}")
                st.session_state["prompt_openai"].append(user_input)
                st.session_state["completion_openai"].append(output)

        if st.session_state["completion_openai"]:
            with response_container:
                for i in range(len(st.session_state["completion_openai"])):
                    message(
                        st.session_state["prompt_openai"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )
                    message(st.session_state["completion_openai"][i], key=str(i))

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

def page_repajama():
    def callback_reset():
        print("callback_reset")
        st.session_state["chat_start_redpajama"] = False
        st.session_state["chat_mode_redpajama"] = CHAT_ONLY
        st.session_state["chat_model_redpajama"] = REDPAJAMA_CHAT_3B
        st.session_state["chat_gpu_redpajama"] = False
        del st.session_state["chat_bot_redpajama"]
        st.session_state["chat_bot_redpajama"] = None
        st.session_state["prompt_redpajama"] = []
        st.session_state["completion_redpajama"] = []
        gc.collect()

    st.button("Reset Chatbot", on_click=callback_reset)

    def callback():
        print("callback")
        if st.session_state["chat_mode_redpajama"] in st.session_state["chat_mode_redpajama_options"]:
            st.session_state["chat_start_redpajama"] = True

    def home():
        st.markdown(
            """

        Please complete the setting below!

        ##### Model?
        - 3B: a model that can be run with 16GB system memory or at least 8GB VRAM with CUDA
        - 7B: a model that can be run with 32GB system memory or at least 16GB VRAM with CUDA

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

        st.selectbox("Choose a LLM model:", st.session_state["chat_model_redpajama_options"], key=persist("chat_model_redpajama"))
        # Add a checkbox to the second column

        st.checkbox("CUDA", key=persist("chat_gpu_redpajama"))
        st.radio("Choose a chat mode:", st.session_state["chat_mode_redpajama_options"], key=persist("chat_mode_redpajama"))
        start = st.button("Start", on_click=callback)
        if start:
            print("start clicked")

    def run():
        # container for chat history
        response_container = st.container()
        # container for text box
        container = st.container()

        chatbot = st.session_state["chat_bot_redpajama"]

        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_area("You:", key="input", height=100)
                submit_button = st.form_submit_button(label="Send")

            if submit_button and user_input:
                chatbot.user_input(prompt=user_input)
                output = chatbot.bot_response()
                print(f"output: {output}")
                st.session_state["prompt_redpajama"].append(user_input)
                st.session_state["completion_redpajama"].append(output)

        if st.session_state["completion_redpajama"]:
            with response_container:
                for i in range(len(st.session_state["completion_redpajama"])):
                    message(
                        st.session_state["prompt_redpajama"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )
                    message(st.session_state["completion_redpajama"][i], key=str(i))

    if "chat_start_redpajama" not in st.session_state:
        if st.session_state["chat_bot_redpajama"] is not None:
            st.session_state["chat_start_redpajama"] = True
        else:
            st.session_state["chat_start_redpajama"] = False
            
    if st.session_state["chat_start_redpajama"] == False:
        print("Welcome to Redpajama Chatbot")
        home()
    else:
        print("Starting Redpajama Chatbot")
        if st.session_state["chat_mode_redpajama"] is not None:
            print(st.session_state["chat_mode_redpajama"])
            # try to load model
            if (
                st.session_state["chat_mode_redpajama"] == CHAT_ONLY
                and st.session_state["chat_bot_redpajama"] is None
            ):
                st.session_state["chat_bot_redpajama"] = RedpajamaChatBotBase(model=st.session_state["chat_model_redpajama"], gpu= st.session_state["chat_gpu_redpajama"] ,gui_mode=True)
                print(f"Redpajama Chatbot: {CHAT_ONLY}")
            elif (st.session_state["chat_mode_redpajama"] == OPEN
                and st.session_state["chat_bot_redpajama"] is None
            ):
                st.session_state["chat_bot_redpajama"] = RedpajamaChatBot(model=st.session_state["chat_model_redpajama"], gpu= st.session_state["chat_gpu_redpajama"], gui_mode=True)
                print(f"Redpajama Chatbot: {OPEN}")
            else:
                if st.session_state["chat_bot_redpajama"] is None:
                    st.session_state["chat_bot_redpajama"] = RedpajamaChatBotBase(model=st.session_state["chat_model_redpajama"], gpu= st.session_state["chat_gpu_redpajama"], gui_mode=True)
            run()
        else:
            home()

PAGES = {
    "home": page_home,
    "Status": page_settings,
    "azure-chatbot": page_azure,
    "openai-chatbot": page_openai,
    "redpajama-chatbot": page_repajama,
}

if __name__ == "__main__":
    load_widget_state()
    main()
