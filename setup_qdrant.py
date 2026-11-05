
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "chatbot_documents"

if not client.collection_exists(COLLECTION_NAME):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

    print("Collection created successfully!")

else:
    print("Collection already exists.")

# Verify collection
info = client.get_collection(COLLECTION_NAME)

print("Collection:", COLLECTION_NAME)
print("Vector configuration:", info.config.params.vectors)

client.close()
