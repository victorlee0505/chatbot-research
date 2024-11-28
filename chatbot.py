
from chatbot_research.huggingface.chatbots.hf_chatbot_base import HuggingFaceChatBotBase
from chatbot_research.huggingface.chatbots.hf_chatbot_chroma import HuggingFaceChatBotChroma
# from chatbot_research.huggingface.chatbots.hf_chatbot_chroma_multi import HuggingFaceChatBotChromaPdr
from chatbot_research.huggingface.chatbots.hf_chatbot_faiss import HuggingFaceChatBotFaiss
from chatbot_research.huggingface.config.hf_llm_config import (
    OPENORCA_MISTRAL_8K_7B,
    OPENORCA_MISTRAL_7B_Q5,
    HERMES_3_LLAMA_3_1_8B_Q8,
)

def main():

    # bot = HuggingFaceChatBotBase(llm_config=HERMES_3_LLAMA_3_1_8B_Q8, disable_mem=True)
    bot = HuggingFaceChatBotFaiss(llm_config=HERMES_3_LLAMA_3_1_8B_Q8, disable_mem=True)

    ## old ##
    # bot = HuggingFaceChatBotBase(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=OPENORCA_MISTRAL_7B_Q5, disable_mem=True)
    
    # bot = HuggingFaceChatBotFaiss(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)
    # bot = HuggingFaceChatBotFaiss(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)

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

if __name__ == "__main__":
    main()
