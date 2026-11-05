
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Qdrant URL or API key is missing from .env")

# Connect to Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=20
)

try:
    # Check connection
    collections = client.get_collections()

    print("Successfully connected to Qdrant Cloud!")
    print("Existing collections:")

    for collection in collections.collections:
        print("-", collection.name)

except Exception as error:
    print("Connection failed:", type(error).__name__)
    raise

finally:
    client.close()
