from langchain_core.prompts import ChatPromptTemplate

class MistralOpenorcaPrompt:

    mistral_openorca_prompt = """
        <|im_start|>user
        {input}<|im_end|>
        <|im_start|>assistant
        """

    # Shared base template for memory
    BASE_MEMORY_TEMPLATE = """
    <|im_start|>system
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    <|im_end|>

    {placeholder}
    """

    # Shared base template for no memory
    BASE_NO_MEMORY_TEMPLATE = """
    <|im_start|>system
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    <|im_end|>
    {placeholder}
    """

    # Shared base template for QA
    BASE_QA_TEMPLATE = """
    <|im_start|>system
    Use ONLY the context provided to answer the question at the end.
    If the context is not relevant, DO NOT try to use your own knowledge and simply say you don't know.

    ### context: {context}
    <|im_end|>
    {placeholder}
    """

    # Mistral-specific memory template
    MEMORY_PROMPT_TEMPLATE = BASE_MEMORY_TEMPLATE.replace(
        "{placeholder}", mistral_openorca_prompt
    )
    MEMORY_PROMPT = ChatPromptTemplate.from_template(template=MEMORY_PROMPT_TEMPLATE)

    # Mistral-specific no memory template
    NO_MEMORY_PROMPT_TEMPLATE = BASE_NO_MEMORY_TEMPLATE.replace(
        "{placeholder}", mistral_openorca_prompt
    )
    NO_MEMORY_PROMPT = ChatPromptTemplate.from_template(template=NO_MEMORY_PROMPT_TEMPLATE)

    # Mistral-specific QA template
    QA_PROMPT_TEMPLATE = BASE_QA_TEMPLATE.replace("{placeholder}", mistral_openorca_prompt)
    QA_PROMPT = ChatPromptTemplate.from_template(template=QA_PROMPT_TEMPLATE)
