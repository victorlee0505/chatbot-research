from chatbot_research.huggingface.chatbots.hf_chatbot_form import HuggingFaceChat2Form
from chatbot_research.huggingface.config.hf_llm import HuggingFaceLLM
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
    GLAIVEAI_FUNC_CALL_3B,
)

llm = HuggingFaceLLM(llm_config=GLAIVEAI_FUNC_CALL_3B, disable_mem=True)

# start chatting
processor = HuggingFaceChat2Form(llm=llm)
processor.process_info()