from langchain_core.prompts import ChatPromptTemplate

# Chatbot Base
memory_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}

{placeholder}
"""

no_mem_template = """
You are a talkative and creative AI writer and provides lots of specific details from its context to answer the following 

{placeholder}
"""

#Chroma
condensed_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

qa_prompt_template = """
Use ONLY the context provided to answer the question at the end.
If the context is not relevant, DO NOT try to use your own knowledge and simply say you don't know. 

{context}

{placeholder}
"""

redpajama_prompt = """
<human>: {input}
<bot>:"""

vicuna_prompt = """
USER: {input}
ASSISTANT:"""

falcon_prompt = """
User: {input}
Assistant:"""

mistral_openorca_prompt = """
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
"""

redpajama_template = memory_template.replace("{placeholder}", redpajama_prompt)
REDPAJAMA_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=redpajama_template)
redpajama_no_mem_template = no_mem_template.replace("{placeholder}", redpajama_prompt)
REDPAJAMA_NO_MEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=redpajama_no_mem_template)
redpajama_qa_template = qa_prompt_template.replace("{placeholder}", redpajama_prompt)
redpajama_qa_template = redpajama_qa_template.replace("{input}", "{question}")
REDPAJAMA_QA_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=redpajama_qa_template)

vicuna_template = memory_template.replace("{placeholder}", vicuna_prompt)
VICUNA_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=vicuna_template)
vicuna_no_mem_template = no_mem_template.replace("{placeholder}", vicuna_prompt)
VICUNA_NO_MEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=vicuna_no_mem_template)
vicuna_qa_template = qa_prompt_template.replace("{placeholder}", vicuna_prompt)
vicuna_qa_template = vicuna_qa_template.replace("{input}", "{question}")
VICUNA_QA_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=vicuna_qa_template)

falcon_template = memory_template.replace("{placeholder}", falcon_prompt)
FALCON_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=falcon_template)
falcon_no_mem_template = no_mem_template.replace("{placeholder}", falcon_prompt)
FALCON_NO_MEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=falcon_no_mem_template)
falcon_qa_template = qa_prompt_template.replace("{placeholder}", falcon_prompt)
falcon_qa_template = falcon_qa_template.replace("{input}", "{question}")
FALCON_QA_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=falcon_qa_template)

wizard_coder_prompt = """
### Instruction:\n{input}\n\n
### Response:"""

wizard_coder_prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
{placeholder}
"""
wizard_coder_template = wizard_coder_prompt_template.replace("{placeholder}", wizard_coder_prompt)
WIZARD_CODER_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=wizard_coder_template)

mistral_prompt_template = """
<|im_start|>system
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
<|im_end|>

{placeholder}
"""

mistral_no_mem_prompt_template = """
<|im_start|>system
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
<|im_end|>
{placeholder}
"""
mistral_qa_prompt_template = """
<|im_start|>system
Use ONLY the context provided to answer the question at the end.
If the context is not relevant, DO NOT try to use your own knowledge and simply say you don't know. 

{context}
<|im_end|>
{placeholder}
"""
mistral_template = mistral_prompt_template.replace("{placeholder}", mistral_openorca_prompt)
MISTRAL_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=mistral_template)
mistral_no_mem_template = mistral_no_mem_prompt_template.replace("{placeholder}", mistral_openorca_prompt)
MISTRAL_NO_MEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=mistral_no_mem_template)
mistral_qa_template = mistral_qa_prompt_template.replace("{placeholder}", mistral_openorca_prompt)
# mistral_qa_template = mistral_qa_template.replace("{input}", "{question}")
MISTRAL_QA_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=mistral_qa_template)

starchat_prompt = "<|system|> Below is a conversation between a human user and a helpful AI coding assistant. <|end|>\n<|user|> {input} <|end|>\n<|assistant|>"
starchat_template = memory_template.replace("{placeholder}", starchat_prompt)
STARCHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=starchat_template)
STARCHAT_NO_MEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=starchat_prompt)
# STARCHAT DO NOY HAVE QA PROMPT

template = """
You are a talkative and creative AI writer and provides lots of specific details from its context to answer the following 

Question: {input}

Helpful Answer:"""

NO_MEM_PROMPT = ChatPromptTemplate.from_template(template=template)

STRAIGHT_PROMPT = ChatPromptTemplate.from_template(template="{input}")

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
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
QA_PROMPT_DOCUMENT_CHAT = ChatPromptTemplate.from_template(
    template=prompt_template
)

func_call_template = """
SYSTEM: You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed

{placeholder}
"""

func_call_prompt = """
USER: {input}
"""

func_call_no_mem_template = func_call_template.replace("{placeholder}", func_call_prompt)
FUNC_CALL_NO_MEM_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(template=func_call_no_mem_template)

# # Llama style (with no system message)\
system_prompt = ''
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "", ""

TRELIS_FUNC_CALL_PROMPT = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{input} {E_INST}\n\n"