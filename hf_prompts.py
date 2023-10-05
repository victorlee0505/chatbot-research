from langchain.prompts import PromptTemplate

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
REDPAJAMA_PROMPT_TEMPLATE = PromptTemplate(input_variables=["history", "input"], template=redpajama_template)
redpajama_no_mem_template = no_mem_template.replace("{placeholder}", redpajama_prompt)
REDPAJAMA_NO_MEM_PROMPT_TEMPLATE = PromptTemplate(input_variables=["input"], template=redpajama_no_mem_template)
redpajama_qa_template = qa_prompt_template.replace("{placeholder}", redpajama_prompt)
REDPAJAMA_QA_PROMPT_TEMPLATE = PromptTemplate(input_variables=["context", "input"], template=redpajama_qa_template)

vicuna_template = memory_template.replace("{placeholder}", vicuna_prompt)
VICUNA_PROMPT_TEMPLATE = PromptTemplate(input_variables=["history", "input"], template=vicuna_template)
vicuna_no_mem_template = no_mem_template.replace("{placeholder}", vicuna_prompt)
VICUNA_NO_MEM_PROMPT_TEMPLATE = PromptTemplate(input_variables=["input"], template=vicuna_no_mem_template)
vicuna_qa_template = qa_prompt_template.replace("{placeholder}", vicuna_prompt)
VICUNA_QA_PROMPT_TEMPLATE = PromptTemplate(input_variables=["context", "input"], template=vicuna_qa_template)

falcon_template = memory_template.replace("{placeholder}", falcon_prompt)
FALCON_PROMPT_TEMPLATE = PromptTemplate(input_variables=["history", "input"], template=falcon_template)
falcon_no_mem_template = no_mem_template.replace("{placeholder}", falcon_prompt)
FALCON_NO_MEM_PROMPT_TEMPLATE = PromptTemplate(input_variables=["input"], template=falcon_no_mem_template)
falcon_qa_template = qa_prompt_template.replace("{placeholder}", falcon_prompt)
FALCON_QA_PROMPT_TEMPLATE = PromptTemplate(input_variables=["context", "input"], template=falcon_qa_template)

wizard_coder_prompt = """
### Instruction:\n{input}\n\n
### Response:"""

wizard_coder_prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
{placeholder}
"""
wizard_coder_template = wizard_coder_prompt_template.replace("{placeholder}", wizard_coder_prompt)
WIZARD_CODER_PROMPT_TEMPLATE = PromptTemplate(input_variables=["input"], template=wizard_coder_template)

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
qa_prompt_template = """
<|im_start|>system
Use ONLY the context provided to answer the question at the end.
If the context is not relevant, DO NOT try to use your own knowledge and simply say you don't know. 

{context}
<|im_end|>
{placeholder}
"""
mistral_template = mistral_prompt_template.replace("{placeholder}", mistral_openorca_prompt)
MISTRAL_PROMPT_TEMPLATE = PromptTemplate(input_variables=["history", "input"], template=mistral_template)
mistral_no_mem_template = mistral_no_mem_prompt_template.replace("{placeholder}", mistral_openorca_prompt)
MISTRAL_NO_MEM_PROMPT_TEMPLATE = PromptTemplate(input_variables=["input"], template=mistral_no_mem_template)
mistral_qa_template = qa_prompt_template.replace("{placeholder}", mistral_openorca_prompt)
MISTRAL_QA_PROMPT_TEMPLATE = PromptTemplate(input_variables=["context", "input"], template=mistral_qa_template)

starchat_prompt = "<|system|> Below is a conversation between a human user and a helpful AI coding assistant. <|end|>\n<|user|> {input} <|end|>\n<|assistant|>"
starchat_template = memory_template.replace("{placeholder}", starchat_prompt)
STARCHAT_PROMPT_TEMPLATE = PromptTemplate(input_variables=["history", "input"], template=starchat_template)
STARCHAT_NO_MEM_PROMPT_TEMPLATE = PromptTemplate(input_variables=["input"], template=starchat_prompt)
# STARCHAT DO NOY HAVE QA PROMPT

template = """
You are a talkative and creative AI writer and provides lots of specific details from its context to answer the following 

Question: {input}

Helpful Answer:"""

NO_MEM_PROMPT = PromptTemplate(template=template, input_variables=["input"])

STRAIGHT_PROMPT = PromptTemplate(template="{input}", input_variables=["input"])

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
