# Modified from https://github.com/THUDM/ChatGLM-6B/blob/main/api.py
# Modified from https://github.com/opencopilotdev/opencopilot

import asyncio

from threading import Lock, Thread

import uvicorn
from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import StreamingResponse

from entities import (
    GenerateStreamRequest,
    TokenizeRequest,
    TokenizeResponse,
    StreamRequest,
    )
from hf_chatbot_base import HuggingFaceChatBotBase
from hf_llm_config import (
    LMSYS_LONGCHAT_1_5_32K_7B,
    LMSYS_VICUNA_1_5_7B,
    LMSYS_VICUNA_1_5_7B_Q8,
    LMSYS_VICUNA_1_5_13B_Q6,
    LMSYS_VICUNA_1_5_16K_7B,
    LMSYS_VICUNA_1_5_16K_7B_Q8,
    LMSYS_VICUNA_1_5_16K_13B_Q6,
    REDPAJAMA_3B,
    REDPAJAMA_7B,
    STARCHAT_BETA_16B_Q5,
    VICUNA_7B,
    WIZARDLM_FALCON_40B_Q6K,
    LLMConfig
    )

router = APIRouter()

llama_lock = Lock()

def _get_llama():
    try:
        llama_lock.acquire()
        yield llm
    except:
        return None
    finally:
        llama_lock.release()


def create_app(hf_llm: HuggingFaceChatBotBase) -> FastAPI:
    global llm
    llm = hf_llm


    app = FastAPI(
        title="HuggingFace LLM API",
        summary=llm.llm_config.model,
        version="0.0.1",
    )
    app.include_router(router)
    return app

# def start_server(app: FastAPI):
#     uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

# def stop_server():
#     uvicorn.Server.shutdown()

@router.get("/")
async def index():
    return {"title": "OSS LLM API"}


@router.post("/generate")
async def generate(
    request: GenerateStreamRequest, llama: HuggingFaceChatBotBase = Depends(_get_llama)
):
    llama.user_input(request.query)
    generated = llama.bot_response()

    return {"generated": generated}

@router.post("/generate_stream")
async def generate_stream(request: StreamRequest, llama: HuggingFaceChatBotBase = Depends(_get_llama)):
    return StreamingResponse(text_streamer(request.query, llama), media_type="text/event-stream")

async def text_streamer(prompt, llama: HuggingFaceChatBotBase):
    # Start the generation in a separate thread
    thread = Thread(target=run_generation, args=(prompt, llama))
    thread.start()

    # Use a while loop to continuously yield the generated text
    while True:
        try:
            # This is a blocking call until there's a new chunk of text or a stop signal
            new_text = next(llama.streamer)
            yield new_text
        except StopIteration:
            # If we receive a StopIteration, it means the stream has ended
            break
        await asyncio.sleep(0.5)

def run_generation(prompt, llama: HuggingFaceChatBotBase):
    # llama.pipe(prompt)[0]['generated_text']
    llama.user_input(prompt)
    llama.bot_response()

@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest, llama: HuggingFaceChatBotBase = Depends(_get_llama)):
    return TokenizeResponse(tokens=llama.tokenizer.tokenize(request.text))

if __name__ == '__main__':

    # get config
    # build a ChatBot object
    # bot = HuggingFaceChatBotBase(llm_config=REDPAJAMA_3B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=REDPAJAMA_7B, disable_mem=True)
    # bot = HuggingFaceChatBotBase(llm_config=VICUNA_7B, disable_mem=True)

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_LONGCHAT_1_5_32K_7B)

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B, disable_mem=True)
    bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B, disable_mem=True, server_mode=True)
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_LONGCHAT_1_5_32K_7B, disable_mem=True)

    # GGUF Quantantized LLM, use less RAM
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_7B_Q8, disable_mem=True, gpu_layers=10) # mem = 10GB

    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_13B_Q8, disable_mem=True, gpu_layers=10) # mem = 18GB
    # bot = HuggingFaceChatBotBase(llm_config=LMSYS_VICUNA_1_5_16K_13B_Q8, disable_mem=True, gpu_layers=10) # mem = 18GB

    # bot = HuggingFaceChatBotBase(llm_config=STARCHAT_BETA_16B_Q8, disable_mem=True, gpu_layers=10) # mem = 23GB
    
    # This one is not good at all
    # bot = HuggingFaceChatBotBase(llm_config=WIZARDLM_FALCON_40B_Q6K, disable_mem=True, gpu_layers=10) # mem = 45GB

    app = create_app(bot)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
