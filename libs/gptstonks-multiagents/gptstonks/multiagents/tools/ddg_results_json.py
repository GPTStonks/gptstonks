import json
from typing import Optional

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.callbacks import CallbackManagerForToolRun


class DuckDuckGoSearchResultsJson(DuckDuckGoSearchResults):
    """Extends DuckDuckGoSearchResults to return a JSON format."""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        res = self.api_wrapper.results(query, self.max_results, source=self.backend)
        return json.dumps(res)
