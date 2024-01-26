from pydantic import BaseModel


class QueryIn(BaseModel):
    query: str
