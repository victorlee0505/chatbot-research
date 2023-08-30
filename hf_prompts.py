from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    template=_template, input_variables=["chat_history", "question"]
)

# prompt_template = """Use ONLY the context provided to answer the question at the end.
# If there isn't enough information from the context, say you don't know. Do not generate answers that don't use the context below.
# If you don't know the answer, just say you don't know. DO NOT try to make up an answer.

# {context}

# Question: {question}
# Helpful Answer:"""
prompt_template = """
Use ONLY the context provided to answer the question at the end.
If the context is not relevant, DO NOT try to use your own knowledge and simply say you don't know. 

{context}

Question: {question}
Answer:"""
QA_PROMPT_DOCUMENT_CHAT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
