import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.agents import AgentAction
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, computed_field


class ToolExecutionOrderCallback(BaseModel, AsyncCallbackHandler):
    """Callback to get the agent's tool execution order."""

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


class LLMTimeCallback(BaseModel, AsyncCallbackHandler):
    """Callback to get the agent's LLM execution times."""

    _last_start_time: float | None = None
    llm_executions_times_ns: list[float] = []

    @computed_field
    @property
    def llm_executions_times_seconds(self) -> list[float]:
        return [el / 1e9 for el in self.llm_executions_times_ns]

    @computed_field
    @property
    def total_llm_execution_time_ns(self) -> float:
        return np.sum(self.llm_executions_times_ns)

    @computed_field
    @property
    def total_llm_execution_time_seconds(self) -> float:
        return self.total_llm_execution_time_ns / 1e9

    @computed_field
    @property
    def num_executions(self) -> int:
        return len(self.llm_executions_times_ns)

    @computed_field
    @property
    def average_llm_executin_time_seconds(self) -> float:
        return self.total_llm_execution_time_seconds / self.num_executions

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chat model starts running."""
        self._last_start_time = time.time_ns()

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        end_time = time.time_ns()
        self.llm_executions_times_ns.append(end_time - self._last_start_time)
