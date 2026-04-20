import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

load_dotenv()


class MongoDB:
    def __init__(self):
        self._client: AsyncIOMotorClient | None = None
        self._db_name: str = os.getenv("MONGODB_DB", "lec-project")
        self._collection_name: str = os.getenv("MONGODB_COLLECTION", "embeddings")

    def connect(self) -> None:
        uri = os.environ["MONGODB_CONNECTION_STRING"]
        self._client = AsyncIOMotorClient(uri)

    def get_db(self) -> AsyncIOMotorDatabase:
        if self._client is None:
            self.connect()
        return self._client[self._db_name]

    def get_collection(self, name: str | None = None) -> AsyncIOMotorCollection:
        return self.get_db()[name or self._collection_name]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


mongo = MongoDB()