from typing import Any, List, Optional
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.agents import AgentAction
from pydantic import BaseModel


class ToolExecutionOrderCallback(BaseModel, AsyncCallbackHandler):
    tools_used: list[str] = []

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.tools_used.append(action.tool)
