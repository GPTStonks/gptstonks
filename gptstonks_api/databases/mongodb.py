import os
import traceback

from pymongo import MongoClient
from pymongo.database import Database


def init_mongo_db() -> Database:
    """Connect to MongoDB database."""

    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        return client[os.getenv("MONGO_DBNAME")]
    except Exception:
        raise RuntimeError(f"Error creating MongoDB client. Trace:\n{traceback.format_exc()}")


db = init_mongo_db()
