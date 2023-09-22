from pydantic import BaseModel
from pydantic import Field
from typing import List

class GenerateStreamRequest(BaseModel):
    query: str
    temperature: float = 0.1
    max_tokens: int

class StreamRequest(BaseModel):
    query: str

class TokenizeRequest(BaseModel):
    text: str


class TokenizeResponse(BaseModel):
    tokens: List[int] = Field(description="List of tokens")

    class Config:
        schema_extra = {"example": {"tokens": [1, 52, 332, 44, 16]}}
