from fastapi import APIRouter

from ..databases import db
from ..models import TokenData

router = APIRouter(prefix="/tokens", tags=["tokens"])


@router.post("/", tags=["tokens"])
async def update_token(token_data: TokenData):
    """Update the token used to access OpenBB.

    Args:
        token_data (TokenData): Token data.

    Returns:
        dict: Response to the query.
    """
    db.tokens.update_one({}, {"$set": token_data.dict()}, upsert=True)
    return {"message": "Token updated"}


@router.get("/", tags=["tokens"])
async def get_token():
    """Get the token used to access OpenBB.

    Returns:
        dict: Response to the query.
    """
    token = db.tokens.find_one({}, {"_id": 0, "openbb": 1})
    return token if token else {"openbb": ""}
