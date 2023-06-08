# Chatbot-research
Experimental research project to explore way build chatbot with Azure OpenAI as well as offline with HuggingFace.

I will be using [RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1) for HuggingFace Inference because it is small and efficient. 

Please check out their website to understand the hardware requirement.


# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

I recommend using python 3.9, but you may try 3.10 or 3.11

```shell
pip install -r requirements.txt
```

Please also install pytorch
```shell
pip3 install torch torchvision torchaudio
```
If you wish to utilize CUDA (make sure you have cuda driver installed and have at least 8GB of VRAM or it will crash)

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Environment Variable

For Azure OpenAI
```
OPENAI_API_KEY

OPENAI_AZURE_BASE_URL

OPENAI_DEPLOYMENT_NAME
```
For HuggingFace, nothing.

# How to run
## Just run azure_chatbot.py or azure_chatbot_base.py
```shell
python azure_chatbot.py
python azure_chatbot_base.py
```
- Base = chat only and do not ingest any document.
- it will go to source_document folder and ingest data and persist using chroma (i took it from PrivateGPT).
- It uses OpenAIEmbeddings which also require your `Azure OpenAI Environment Variable`
- it will then initialize the model and start interacting

## Just run hf_redpajama_chatbot.py
```shell
python hf_redpajama_chatbot.py
python hf_redpajama_chatbot_base.py
```
- Base = chat only and do not ingest any document.
- it will go to source_document folder and ingest data and persist using chroma (i took it from PrivateGPT).
- It uses HuggingFaceEmbeddings (require internet to download sentence_transformers model)
- it will then initialize the model (first time require internet to download model) and start interacting

## Just run ingest.py
```shell
python ingest.py
```
- it will go to source_document folder and ingest data and persist using chroma (i took it from PrivateGPT).
- `Ingestion(offline=True)` to switch from online (OpenAIEmbeddings), by default it is offline (HuggingFaceEmbeddings) embedding
- you can run Ingestion first and then run chatbot, chatbot will detect if a vector story already existed. this way you can test how long does it take to ingest.
- OpenAIEmbedded vector store can only be use by azure_chatbot and vice versa.
- ingestion specification is same as [PrivateGPT](https://github.com/imartinez/privateGPT)



# Disclaimer
This is a just for fun project for personal research purpose. It is not production ready, and it is not meant to be used in production. The models selection is not optimized for performance, but for privacy; but it is possible to use different models and vectorstores to improve performance.