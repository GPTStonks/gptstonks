import traceback

from pymongo import MongoClient
from pymongo.database import Database

from ..constants import MONGO_DBNAME, MONGO_URI


def init_mongo_db() -> Database:
    """Connect to MongoDB database."""

    try:
        client = MongoClient(MONGO_URI)
        return client[MONGO_DBNAME]
    except Exception:
        raise RuntimeError(f"Error creating MongoDB client. Trace:\n{traceback.format_exc()}")


db = init_mongo_db()
