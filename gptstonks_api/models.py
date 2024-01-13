from pydantic import BaseModel


class TokenData(BaseModel):
    openbb: str
