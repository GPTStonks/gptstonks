from fastapi import APIRouter

from ..databases import db
from ..models import MessageResponse, TokenData, TokenResponse

router = APIRouter(prefix="/tokens", tags=["tokens"])


@router.post("/", tags=["tokens"])
async def update_token(token_data: TokenData) -> MessageResponse:
    """Update the token used to access OpenBB.

    Args:
        token_data (`TokenData`): Token data information to update the database.

    Returns:
        `MessageResponse`: message indicating success.
    """
    db.tokens.update_one({}, {"$set": token_data.dict()}, upsert=True)
    return MessageResponse(message="Token updated")


@router.get("/", tags=["tokens"])
async def get_token() -> TokenResponse:
    """Get the token used to access OpenBB.

    Returns:
        `TokenResponse`: token for OpenBB.
    """
    token = db.tokens.find_one({}, {"_id": 0, "openbb": 1})
    return TokenResponse.model_validate(token) if token else TokenResponse(openbb="")
