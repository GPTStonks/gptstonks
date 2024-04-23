import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class BaseAgentState(TypedDict):
    """Base state."""

    # input to the agent
    input: str
    # list of messages by tools to use as context
    context_messages: Annotated[list[BaseMessage], operator.add]
    # final response
    response: str
    # extra arguments
    extra: dict[str, Any]
