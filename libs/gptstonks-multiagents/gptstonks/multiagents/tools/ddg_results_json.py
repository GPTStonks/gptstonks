import json
from typing import Literal, Optional

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.callbacks import CallbackManagerForToolRun


class DuckDuckGoSearchResultsJson(DuckDuckGoSearchResults):
    """Extends DuckDuckGoSearchResults to return a JSON format."""

    response_format: Literal["content_and_artifact"] = "content"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        res = self.api_wrapper.results(query, self.max_results, source=self.backend)
        return json.dumps(res)
