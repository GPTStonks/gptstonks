from typing import Any

from langchain.llms import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import Generation, LLMResult


class ChatModelWithLLMIface(BaseLLM):
    """Wrapper model to transform the ChatModel interface to a LLM interface."""

    chat_model: BaseChatModel
    system_message: str = "You write concise and complete answers."

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""
        outputs = []
        for prompt in prompts:
            messages = [SystemMessage(content=self.system_message), HumanMessage(content=prompt)]
            chat_result = self.chat_model._generate(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            outputs.append(
                [Generation(text=chat_gen.text) for chat_gen in chat_result.generations]
            )
        return LLMResult(generations=outputs)

    def _llm_type(self) -> str:
        """Return type of chat model."""
        return self.chat_model._llm_type()
