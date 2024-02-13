from pydantic import BaseModel


class BaseAgentResponse(BaseModel):
    """Model to define the base response parameters."""

    type: str
    body: str


class DataAgentResponse(BaseAgentResponse):
    """Model to define the data response parameters."""

    result_data: list[dict]
