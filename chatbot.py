
from chatbot_research.huggingface.chatbots.hf_chatbot_base import HuggingFaceChatBotBase
from chatbot_research.huggingface.chatbots.hf_chatbot_chroma import HuggingFaceChatBotChroma
from chatbot_research.huggingface.chatbots.hf_chatbot_chroma_multi import HuggingFaceChatBotChromaPdr
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
)


bot = HuggingFaceChatBotBase(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)
# bot = HuggingFaceChatBotChroma(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)
# bot = HuggingFaceChatBotChromaPdr(llm_config=OPENORCA_MISTRAL_8K_7B, disable_mem=True)

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