from langchain_core.prompts import ChatPromptTemplate

class HermesLlama3Prompt:

    llama3_prompt = """
        <|im_start|>user
        {input}<|im_end|>
        <|im_start|>assistant
        """

    # Shared base template for memory
    BASE_MEMORY_TEMPLATE = """
    <|im_start|>system
    You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user.
    If You do not know the answer to a question, you truthfully says you do not know.
    Current conversation:
    {history}
    <|im_end|>

    {placeholder}
    """

    # Shared base template for no memory
    BASE_NO_MEMORY_TEMPLATE = """
    <|im_start|>system
    You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user.
    If You do not know the answer to a question, you truthfully says you do not know. <|im_end|>
    {placeholder}
    """

    # Shared base template for QA
    BASE_QA_TEMPLATE = """
    <|im_start|>system
    You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user.
    Use ONLY the context provided to answer the question at the end.
    If the context is not relevant, DO NOT try to use your own knowledge and simply say you don't know.

    ### context: {context}
    <|im_end|>
    {placeholder}
    """

    # Mistral-specific memory template
    MEMORY_PROMPT_TEMPLATE = BASE_MEMORY_TEMPLATE.replace(
        "{placeholder}", llama3_prompt
    )
    MEMORY_PROMPT = ChatPromptTemplate.from_template(template=MEMORY_PROMPT_TEMPLATE)

    # Mistral-specific no memory template
    NO_MEMORY_PROMPT_TEMPLATE = BASE_NO_MEMORY_TEMPLATE.replace(
        "{placeholder}", llama3_prompt
    )
    NO_MEMORY_PROMPT = ChatPromptTemplate.from_template(template=NO_MEMORY_PROMPT_TEMPLATE)

    # Mistral-specific QA template
    QA_PROMPT_TEMPLATE = BASE_QA_TEMPLATE.replace("{placeholder}", llama3_prompt)
    QA_PROMPT = ChatPromptTemplate.from_template(template=QA_PROMPT_TEMPLATE)
