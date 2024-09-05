# Modified from https://github.com/opencopilotdev/opencopilot
import asyncio
import time
import json
from threading import Lock, Thread
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from urllib.parse import urljoin
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import aiohttp
import requests
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.adapters.openai import convert_dict_to_message, convert_message_to_dict
from langchain.schema import AIMessage
from langchain.schema import BaseMessage
from langchain.schema import ChatGeneration
from langchain.schema import ChatResult
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import Extra
from chatbot_research.huggingface.config.hf_prompts import NO_MEM_PROMPT
from chatbot_research.huggingface.config.hf_llm import HuggingFaceLLM
from chatbot_research.huggingface.chatbots.hf_chatbot_base import HuggingFaceChatBotBase
from chatbot_research.huggingface.chatbots.hf_chatbot_chroma import HuggingFaceChatBotChroma
from chatbot_research.huggingface.utils.hf_streaming_util import run_generation
from transformers import Pipeline

class ChatHuggingFace(BaseChatModel):

    llm: HuggingFaceLLM
    # tokenizer: Any
    # model: Any

    # def __init__(self, llm: HuggingFaceLLM, *args, **kwargs):
    #     self.tokenizer = llm.tokenizer
    #     self.model = llm.model
    #     super().__init__(*args, **kwargs)
    #     # self.llm = llm


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        allow_population_by_field_name = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = ""
        for message in messages:
            prompt += message.content
            prompt += "\n"
        
        final = self.llm.pipeline(prompt)
        additional_kwargs={**kwargs}
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=final, additional_kwargs=additional_kwargs),
                )
            ]
        )
    
    def _convert_dict_to_message(self, _dict: Mapping[str, Any]) -> BaseMessage:
        role = _dict["role"]
        if role == "user":
            return HumanMessage(content=_dict["content"])
        elif role == "assistant":
            # Fix for azure
            # Also OpenAI returns None for tool invocations
            content = _dict.get("content") or ""
            if _dict.get("function_call"):
                _dict["function_call"]["arguments"] = json.dumps(
                    _dict["function_call"]["arguments"]
                )
                additional_kwargs = {"function_call": dict(_dict["function_call"])}
            else:
                additional_kwargs = {}
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif role == "system":
            return SystemMessage(content=_dict["content"])
        elif role == "function":
            return FunctionMessage(content=_dict["content"], name=_dict["name"])
        else:
            return ChatMessage(content=_dict["content"], role=role)
    
    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = self._convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        return ChatResult(generations=generations)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final = ""
        try:
            async for text in self._get_async_stream(
                messages[0],
            ):
                final += text

            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=final, additional_kwargs=kwargs),
                    )
                ]
            )
        except Exception as exc:
            print("LOCAL_LLM_CONNECTION_ERROR")

    def get_token_ids(self, text: str) -> List[int]:
        try:
            return self.llm.pipeline.tokenizer.tokenize(text)
        except Exception as exc:
            print("LOCAL_LLM_CONNECTION_ERROR")

    @property
    def _llm_type(self) -> str:
        return "local-llm"

    def _get_stream(self, prompt):
        thread = Thread(target=run_generation, args=(prompt, self.llm))
        thread.start()
        time.sleep(20)
        # Use a while loop to continuously yield the generated text
        while True:
            try:
                # This is a blocking call until there's a new chunk of text or a stop signal
                new_text = next(self.llm.pipeline.streamer)
                yield new_text
            except StopIteration:
                # If we receive a StopIteration, it means the stream has ended
                break
            # await asyncio.sleep(0.5)

    async def _get_async_stream(self, prompt):
        thread = Thread(target=run_generation, args=(prompt, self.llm))
        thread.start()
        time.sleep(20)
        # Use a while loop to continuously yield the generated text
        while True:
            try:
                # This is a blocking call until there's a new chunk of text or a stop signal
                new_text = next(self.llm.pipeline.streamer)
                yield new_text
            except StopIteration:
                # If we receive a StopIteration, it means the stream has ended
                break
            await asyncio.sleep(0.5)
