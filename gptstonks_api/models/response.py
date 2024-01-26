from pydantic import BaseModel


class BaseAgentResponse(BaseModel):
    type: str
    body: str


class DataAgentResponse(BaseAgentResponse):
    result_data: list[dict]
