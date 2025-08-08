from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings

client = AsyncIOMotorClient(settings.MONGO_URI)
db = client["cognidocs"]

def get_collection(name: str):
    return db[name]
