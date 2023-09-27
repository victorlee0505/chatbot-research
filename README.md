# Chatbot-research
Experimental research project to explore way build chatbot with Azure OpenAI as well as offline with HuggingFace.

I will be using [RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1) for HuggingFace Inference because it is small and efficient. 

Please check out their website to understand the hardware requirement.

All models from HuggingFace will be downloaded first time you run it and stored in C:\Users\[YOUR_USER_NAME]\.cache

If you delete model in .cache (ex. save disk space), model will be re-download next time you start up the chatbot.


# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

I recommend using python 3.10 (best support for torch), but you may try 3.9 or 3.11

```shell
pip3 install torch torchvision torchaudio
```
If you wish to utilize CUDA (make sure you have cuda driver installed and have at least 8GB of VRAM or it will crash)

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```shell
pip install -r requirements.txt
```

# Environment Variable

For Azure OpenAI
```
AZURE_OPENAI_API_KEY

AZURE_OPENAI_BASE_URL

AZURE_OPENAI_DEPLOYMENT_NAME
```

For OpenAI
```
OPENAI_API_KEY

```

For HuggingFace, nothing.

# How to run

### `Although start up chroma chatbot will auto ingest (from default path ./source_documents), I highy recommand you to do the ingest separately where you can choose to use OpenAI sbuscription, enable private local offline embedding, override the source path and enable CUDA for ingestion`

## Just run ingest.py
```shell
python ingest.py
```
- it will go to source_document folder and ingest data and persist using chroma (i took it from PrivateGPT).
- `Ingestion(offline=True)` Default False: online (OpenAIEmbeddings), offline=True to use offline (HuggingFaceEmbeddings) embedding
- `Ingestion(offline=True, gpu=True, source_path=base_path)` to toggle CUDA
- `Ingestion(offline=True, gpu=True, source_path=base_path)` to toggle CUDA and override source path
- `Ingestion(openai=True)` to use OpenAI subscription for embedding
- you can run Ingestion first and then run chatbot, chatbot will detect if a vector story already existed. this way you can test how long does it take to ingest.
- OpenAIEmbedded vector store can only be use by azure_chatbot and vice versa.
- ingestion specification is same as [PrivateGPT](https://github.com/imartinez/privateGPT)

### `delete ./storage_azure or ./storage_hf will remove vector storage so you can start over`
## Just run azure_chatbot.py or azure_chatbot_base.py
```shell
python azure_chatbot_base.py
python azure_chatbot.py
```
- azure_chatbot_base = chat only like ChatGPT.
- azure_chatbot.py = chat and ingest like PrivateGPT.
- it will go to source_document folder and ingest data and persist using chroma (i took it from PrivateGPT).
- if chroma db is found, DATA will not be reloaded.
- It uses OpenAIEmbeddings which also require your `Azure OpenAI Environment Variable`

## Just run openai_chatbot.py
```shell
python openai_chatbot.py
```
- openai_chatbot = chat and ingest like PrivateGPT.
- The only chat mode is CLOSED so it does not answer anything outside of the context. (don't know how to make it open-ended)
- it will go to source_document folder and ingest data and persist using chroma (i took it from PrivateGPT).
- if chroma db is found, DATA will not be reloaded.
- It uses OpenAIEmbeddings which also require your `OpenAI_API_KEY Environment Variable`

## Just run hf_chatbot_*.py
### Models are downloaded from HuggingFace, so it will take a while to download the first time you run it.
### If you delete the model in .cache, it will re-download next time you run it.
### Please check LLMConfig.py for more details. 
### Running LLM locally require high hardware requriement.
```shell
python hf_chatbot_base.py
python hf_chatbot_chroma.py
python hf_chatbot_coder.py
```
- hf_chatbot_base = chat only and do not ingest any document.
- hf_chatbot_chroma = chat and ingest like PrivateGPT.
- hf_chatbot_coder = chat only with model specialized for coding.
- it will go to source_document folder and ingest data and persist using chroma (i took it from PrivateGPT).
- if chroma db is found, DATA will not be reloaded.
- It uses HuggingFaceEmbeddings (require internet to download sentence_transformers model)
- it will then initialize the model (first time require internet to download model) and start interacting

## GUI
### Streamlit-chat is used for GUI interface for easy interaction
### allow you to choose between Azure OpenAI, OpenAI, HuggingFace
### allow you to choose between different LLM from HuggingFace
```shell
streamlit run app.py
```

## FastAPI
### FastAPI is used for API interface for easy interaction
### allow you to choose between different LLM from HuggingFace
```shell
python api.py
```

# Disclaimer
This is a just for fun project for personal research purpose. It is not production ready, and it is not meant to be used in production. The models selection is not optimized for performance, but for privacy; but it is possible to use different models and vectorstores to improve performance.