from langchain.agents.agent import AgentExecutor
from langchain_community.utilities import PythonREPL
from pydantic import BaseModel, ConfigDict


class TokenData(BaseModel):
    openbb: str


class AppData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_executor: AgentExecutor | None = None
    python_repl_utility: PythonREPL | None = None
