from pydantic import BaseModel


class QueryIn(BaseModel):
    """Model to define the main query parameters."""

    query: str
