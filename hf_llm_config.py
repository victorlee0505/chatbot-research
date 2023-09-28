from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.chains.conversation.prompt import PROMPT

from hf_prompts import (
    FALCON_NO_MEM_PROMPT_TEMPLATE,
    FALCON_PROMPT_TEMPLATE,
    FALCON_QA_PROMPT_TEMPLATE,
    NO_MEM_PROMPT,
    QA_PROMPT_DOCUMENT_CHAT,
    REDPAJAMA_NO_MEM_PROMPT_TEMPLATE,
    REDPAJAMA_PROMPT_TEMPLATE,
    REDPAJAMA_QA_PROMPT_TEMPLATE,
    STARCHAT_PROMPT_TEMPLATE,
    STARCHAT_NO_MEM_PROMPT_TEMPLATE,
    STRAIGHT_PROMPT,
    VICUNA_NO_MEM_PROMPT_TEMPLATE,
    VICUNA_PROMPT_TEMPLATE,
    VICUNA_QA_PROMPT_TEMPLATE,
    WIZARD_CODER_PROMPT_TEMPLATE,
)


class LLMConfig:
    def __init__(
        self,
        model: str = None,
        model_file: str = None,
        model_type: str = None,
        ai_prefix: str = None,
        human_prefix: str = None,
        prompt_template: BasePromptTemplate = PROMPT,
        prompt_no_mem_template: BasePromptTemplate = NO_MEM_PROMPT,
        prompt_qa_template: BasePromptTemplate = QA_PROMPT_DOCUMENT_CHAT,
        stop_words: list = None,
        eos_token_id: list = None,
        model_max_length: int = 2048,
        max_new_tokens: int = 500,
        max_mem_tokens: int = 500,
        temperature: int = 0.7,
        top_p: int = 0.7,
        top_k: int = 50,
        do_sample: bool = True,  # False: This generally results in more coherent but less diverse output
        retriever_max_tokens_limit: int = 1000,
        target_source_chunks: int = 4,
    ):
        self.model = model
        self.model_file = model_file
        self.model_type = model_type
        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        self.prompt_template = prompt_template
        self.prompt_no_mem_template = prompt_no_mem_template
        self.prompt_qa_template = prompt_qa_template
        self.stop_words = stop_words if stop_words is not None else []
        self.eos_token_id = eos_token_id if eos_token_id is not None else None
        self.model_max_length = model_max_length
        self.max_new_tokens = max_new_tokens
        self.max_mem_tokens = max_mem_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.retriever_max_tokens_limit = retriever_max_tokens_limit
        self.target_source_chunks = target_source_chunks

    def validate(self):
        if not self.model:
            raise ValueError("model is not set.")
        if not self.ai_prefix:
            raise ValueError("ai_prefix is not set.")
        if not self.human_prefix:
            raise ValueError("human_prefix is not set.")


REDPAJAMA_3B = LLMConfig(
    model="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    ai_prefix="<bot>:",
    human_prefix="<human>:",
    prompt_template=REDPAJAMA_PROMPT_TEMPLATE,
    prompt_no_mem_template=REDPAJAMA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=REDPAJAMA_QA_PROMPT_TEMPLATE,
    stop_words=["<human>:", "Question:"],
)

REDPAJAMA_7B = LLMConfig(
    model="togethercomputer/RedPajama-INCITE-7B-Chat",
    ai_prefix="<bot>:",
    human_prefix="<human>:",
    prompt_template=REDPAJAMA_PROMPT_TEMPLATE,
    prompt_no_mem_template=REDPAJAMA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=REDPAJAMA_QA_PROMPT_TEMPLATE,
    stop_words=["<human>:", "Question:"],
)

# LLAMA 1
VICUNA_7B = LLMConfig(
    model="TheBloke/Wizard-Vicuna-7B-Uncensored-HF",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
)

# LLAMA 2
LMSYS_VICUNA_1_5_7B = LLMConfig(
    model="lmsys/vicuna-7b-v1.5",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
    model_max_length=4096,
    max_new_tokens=2000,
    max_mem_tokens=600,
)

LMSYS_VICUNA_1_5_16K_7B = LLMConfig(
    model="lmsys/vicuna-7b-v1.5-16k",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
    model_max_length=16000,
    max_new_tokens=10000,
    max_mem_tokens=2000,
)

LMSYS_LONGCHAT_1_5_32K_7B = LLMConfig(
    model="lmsys/longchat-7b-v1.5-32k",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
    model_max_length=32000,
    max_new_tokens=22000,
    max_mem_tokens=2000,
)

# LLAMA 2 GGUF

LMSYS_VICUNA_1_5_7B_Q8 = LLMConfig(
    model="TheBloke/vicuna-7B-v1.5-GGUF",
    model_file="vicuna-7b-v1.5.Q8_0.gguf",
    model_type="llama",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
    model_max_length=4096,
    max_new_tokens=2000,
    max_mem_tokens=600,
)

LMSYS_VICUNA_1_5_16K_7B_Q8 = LLMConfig(
    model="TheBloke/vicuna-7B-v1.5-16K-GGUF",
    model_file="vicuna-7b-v1.5-16k.Q8_0.gguf",
    model_type="llama",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
    model_max_length=16000,
    max_new_tokens=10000,
    max_mem_tokens=2000,
)

LMSYS_VICUNA_1_5_13B_Q6 = LLMConfig(
    model="TheBloke/vicuna-13B-v1.5-GGUF",
    model_file="vicuna-13b-v1.5.Q6_K.gguf",
    model_type="llama",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
    model_max_length=4096,
    max_new_tokens=2000,
    max_mem_tokens=600,
)

LMSYS_VICUNA_1_5_16K_13B_Q6 = LLMConfig(
    model="TheBloke/vicuna-13B-v1.5-16K-GGUF",
    model_file="vicuna-13b-v1.5-16k.Q6_K.gguf",
    model_type="llama",
    ai_prefix="ASSISTANT:",
    human_prefix="USER:",
    prompt_template=VICUNA_PROMPT_TEMPLATE,
    prompt_no_mem_template=VICUNA_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=VICUNA_QA_PROMPT_TEMPLATE,
    model_max_length=16000,
    max_new_tokens=10000,
    max_mem_tokens=2000,
)

WIZARDLM_FALCON_40B_Q6K = LLMConfig(
    model="TheBloke/WizardLM-Uncensored-Falcon-40B-GGML",
    model_file="wizardlm-uncensored-falcon-40b.ggccv1.q6_k.bin",
    model_type="falcon",
    ai_prefix="Assistant:",
    human_prefix="User:",
    prompt_template=FALCON_PROMPT_TEMPLATE,
    prompt_no_mem_template=FALCON_NO_MEM_PROMPT_TEMPLATE,
    prompt_qa_template=FALCON_QA_PROMPT_TEMPLATE,
    model_max_length=4096,
    max_new_tokens=2000,
    max_mem_tokens=600,
    eos_token_id=["<|endoftext|>"],
)


SANTA_CODER_1B = LLMConfig(
    model="bigcode/santacoder",
    ai_prefix="<|assistant|>",
    human_prefix="<|user|>",
    prompt_template=STRAIGHT_PROMPT,
    prompt_no_mem_template=STRAIGHT_PROMPT,
    prompt_qa_template=STRAIGHT_PROMPT,
    model_max_length=1024,
    temperature=0.1,
)

CODEGEN2_1B = LLMConfig(
    model="Salesforce/codegen2-1B",
    ai_prefix="<|assistant|>",
    human_prefix="<|user|>",
    prompt_template=STRAIGHT_PROMPT,
    prompt_no_mem_template=STRAIGHT_PROMPT,
    prompt_qa_template=STRAIGHT_PROMPT,
    temperature=0.1,
)

CODEGEN2_4B = LLMConfig(
    model="Salesforce/codegen2-3_7B",
    ai_prefix="<|assistant|>",
    human_prefix="<|user|>",
    prompt_template=STRAIGHT_PROMPT,
    prompt_no_mem_template=STRAIGHT_PROMPT,
    prompt_qa_template=STRAIGHT_PROMPT,
    temperature=0.1,
)

CODEGEN25_7B = LLMConfig(
    model="Salesforce/codegen25-7b-multi",
    ai_prefix="<|assistant|>",
    human_prefix="<|user|>",
    prompt_template=STRAIGHT_PROMPT,
    prompt_no_mem_template=STRAIGHT_PROMPT,
    prompt_qa_template=STRAIGHT_PROMPT,
    temperature=0.1,
    max_new_tokens=128,
)

WIZARDCODER_3B = LLMConfig(
    model="WizardLM/WizardCoder-3B-V1.0",
    ai_prefix="Response:",
    human_prefix="Instruction:",
    prompt_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_no_mem_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_qa_template=WIZARD_CODER_PROMPT_TEMPLATE,
    model_max_length=8192,
    max_new_tokens=2048,
    max_mem_tokens=600,
    temperature=1,
    do_sample=False,
    top_p=1,
    stop_words=["</s>"],
    eos_token_id=["</s>"],
)

WIZARDCODER_15B_Q8 = LLMConfig(
    model="TheBloke/WizardCoder-15B-1.0-GGML",
    model_file="WizardCoder-15B-1.0.ggmlv3.q8_0.bin",
    model_type="gpt_bigcode",
    ai_prefix="Response:",
    human_prefix="Instruction:",
    prompt_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_no_mem_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_qa_template=WIZARD_CODER_PROMPT_TEMPLATE,
    model_max_length=8192,
    max_new_tokens=2048,
    max_mem_tokens=600,
    temperature=1,
    do_sample=False,
    top_p=1,
    stop_words=["</s>"],
    eos_token_id=["</s>"],
)

WIZARDCODER_PY_7B = LLMConfig(
    model="WizardLM/WizardCoder-Python-7B-V1.0",
    ai_prefix="Response:",
    human_prefix="Instruction:",
    prompt_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_no_mem_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_qa_template=WIZARD_CODER_PROMPT_TEMPLATE,
    model_max_length=8192,
    max_new_tokens=2048,
    max_mem_tokens=600,
    temperature=1,
    do_sample=False,
    top_p=1,
    stop_words=["</s>"],
    eos_token_id=["</s>"],
)

WIZARDCODER_PY_7B_Q6 = LLMConfig(
    model="TheBloke/WizardCoder-Python-7B-V1.0-GGUF",
    model_file="wizardcoder-python-7b-v1.0.Q6_K.gguf",
    model_type="gpt_bigcode",
    ai_prefix="Response:",
    human_prefix="Instruction:",
    prompt_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_no_mem_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_qa_template=WIZARD_CODER_PROMPT_TEMPLATE,
    model_max_length=8192,
    max_new_tokens=2048,
    max_mem_tokens=600,
    temperature=1,
    do_sample=False,
    top_p=1,
    stop_words=["</s>"],
    eos_token_id=["</s>"],
)

WIZARDCODER_PY_13B_Q6 = LLMConfig(
    model="TheBloke/WizardCoder-Python-13B-V1.0-GGUF",
    model_file="wizardcoder-python-13b-v1.0.Q6_K.gguf",
    model_type="gpt_bigcode",
    ai_prefix="Response:",
    human_prefix="Instruction:",
    prompt_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_no_mem_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_qa_template=WIZARD_CODER_PROMPT_TEMPLATE,
    model_max_length=8192,
    max_new_tokens=2048,
    max_mem_tokens=600,
    temperature=1,
    do_sample=False,
    top_p=1,
    stop_words=["</s>"],
    eos_token_id=["</s>"],
)

WIZARDCODER_PY_34B_Q5 = LLMConfig(
    model="TheBloke/WizardCoder-Python-34B-V1.0-GGUF",
    model_file="wizardcoder-python-34b-v1.0.Q5_K_M.gguf",
    model_type="gpt_bigcode",
    ai_prefix="Response:",
    human_prefix="Instruction:",
    prompt_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_no_mem_template=WIZARD_CODER_PROMPT_TEMPLATE,
    prompt_qa_template=WIZARD_CODER_PROMPT_TEMPLATE,
    model_max_length=8192,
    max_new_tokens=2048,
    max_mem_tokens=600,
    temperature=1,
    do_sample=False,
    top_p=1,
    stop_words=["</s>"],
    eos_token_id=["</s>"],
)

STARCHAT_BETA_16B_Q5 = LLMConfig(
    model="TheBloke/starchat-beta-GGML",
    model_file="starchat-beta.ggmlv3.q5_1.bin",
    model_type="gpt_bigcode",
    ai_prefix="<|assistant|>",
    human_prefix="<|system|> Below is a conversation between a human user and a helpful AI coding assistant. <|end|>\n<|user|>",
    prompt_template=STARCHAT_PROMPT_TEMPLATE,
    prompt_no_mem_template=STARCHAT_NO_MEM_PROMPT_TEMPLATE,
    model_max_length=8192,
    max_new_tokens=2048,
    max_mem_tokens=600,
    temperature=0.2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    stop_words=["<|end|>"],
    eos_token_id=["<|end|>"],
)
