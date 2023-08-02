from langchain import BasePromptTemplate, PromptTemplate
from langchain.chains.conversation.prompt import PROMPT

class LLMConfig:
    def __init__(
        self,
        model: str = None,
        ai_prefix: str = None,
        human_prefix: str = None,
        prompt_template: BasePromptTemplate = PROMPT,
        stop_words: list = None,
        model_max_length: int = 2048,
        max_new_tokens: int = 500,
        temperature: int = 0.7,
        top_p: int = 0.7,
        top_k: int = 50,
        do_sample: bool = True, # False: This generally results in more coherent but less diverse output
        retriever_max_tokens_limit: int = 1000,
        target_source_chunks: int = 4,
    ):
        self.model = model
        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        self.prompt_template = prompt_template
        self.stop_words = stop_words if stop_words is not None else []
        self.model_max_length = model_max_length
        self.max_new_tokens = max_new_tokens
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


REDPAJAMA_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}

<human>: {input}
<bot>:"""

REDPAJAMA_PROMPT_TEMPLATE = PromptTemplate(input_variables=["history", "input"], template=REDPAJAMA_TEMPLATE)

REDPAJAMA_3B = LLMConfig(
    model="togethercomputer/RedPajama-INCITE-Chat-3B-v1", ai_prefix="<bot>:", human_prefix="<human>:", prompt_template=REDPAJAMA_PROMPT_TEMPLATE, stop_words=["<human>:", "Question:"]
)

REDPAJAMA_7B = LLMConfig(
    model="togethercomputer/RedPajama-INCITE-7B-Chat", ai_prefix="<bot>:", human_prefix="<human>:", prompt_template=REDPAJAMA_PROMPT_TEMPLATE, stop_words=["<human>:", "Question:"]
)

VICUNA_7B = LLMConfig(
    model="TheBloke/Wizard-Vicuna-7B-Uncensored-HF", ai_prefix="<bot>:", human_prefix="<human>:"
)
