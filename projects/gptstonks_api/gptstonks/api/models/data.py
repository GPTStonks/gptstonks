from langchain.agents.agent import AgentExecutor
from langchain_experimental.utilities.python import PythonREPL
from pydantic import BaseModel, ConfigDict


class TokenData(BaseModel):
    """Model to define the list of tokens available."""

    openbb: str


class AppData(BaseModel):
    """Model to define the global application data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_executor: AgentExecutor | None = None
    python_repl_utility: PythonREPL | None = None
