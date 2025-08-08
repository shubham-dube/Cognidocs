# core/db.py
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    database: AsyncIOMotorDatabase = None

db_instance = Database()

async def connect_to_mongo():
    """Create database connection and setup indexes."""
    try:
        logger.info("Connecting to MongoDB...")
        db_instance.client = AsyncIOMotorClient(
            settings.MONGO_URI,
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
        )
        
        # Test the connection
        await db_instance.client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        db_instance.database = db_instance.client[settings.MONGO_DB]
        
        # Setup indexes
        await setup_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection."""
    try:
        if db_instance.client:
            db_instance.client.close()
            logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")

async def setup_indexes():
    """Create database indexes for optimal performance."""
    try:
        logger.info("Setting up database indexes...")
        
        # Knowledge Bases collection indexes
        kb_indexes = [
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("kb_type", ASCENDING)]),
            IndexModel([("last_updated", DESCENDING)]),
            IndexModel([("name", "text"), ("description", "text")]),  # Text search
        ]
        
        await db_instance.database.kbs.create_indexes(kb_indexes)
        logger.info("Created indexes for 'kbs' collection")
        
        # You can add more collections and indexes as needed
        # Example: Files collection for tracking individual files
        # file_indexes = [
        #     IndexModel([("kb_id", ASCENDING)]),
        #     IndexModel([("created_at", DESCENDING)]),
        #     IndexModel([("status", ASCENDING)]),
        # ]
        # await db_instance.database.files.create_indexes(file_indexes)
        
        logger.info("All database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error setting up indexes: {e}")
        # Don't raise here as indexes are not critical for basic functionality

def get_db() -> AsyncIOMotorDatabase:
    """Get database instance."""
    if db_instance.database is None:
        raise RuntimeError("Database not initialized. Call connect_to_mongo() first.")
    return db_instance.database

async def get_db_stats():
    """Get database statistics for monitoring."""
    try:
        db = get_db()
        stats = await db.command("dbStats")
        
        # Get collection stats
        collections_stats = {}
        collection_names = await db.list_collection_names()
        
        for collection_name in collection_names:
            collection_stats = await db.command("collStats", collection_name)
            collections_stats[collection_name] = {
                "count": collection_stats.get("count", 0),
                "size": collection_stats.get("size", 0),
                "avgObjSize": collection_stats.get("avgObjSize", 0),
            }
        
        return {
            "database": {
                "collections": stats.get("collections", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0),
                "indexes": stats.get("indexes", 0),
                "indexSize": stats.get("indexSize", 0),
            },
            "collections": collections_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return None

# Health check function
async def check_database_health():
    """Check if database is healthy and responsive."""
    try:
        if not db_instance.client:
            return False, "Database not connected"
        
        # Simple ping test
        await db_instance.client.admin.command('ping')
        
        # Check if we can query a collection
        db = get_db()
        await db.kbs.count_documents({})
        
        return True, "Database healthy"
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False, str(e)