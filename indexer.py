# indexer.py
# Converts parsed functions/files into embeddings and stores them in Pinecone
# Each repo gets its own Pinecone namespace to keep data separate

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "codebase-knowledge-ai"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Load the embedding model once - reused for every embed call
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def ensure_index_exists():
    """
    Creates the Pinecone index if it doesn't already exist.
    all-MiniLM-L6-v2 produces 384-dimensional embeddings.
    """
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )


def build_searchable_text(node_id: str, node_data: dict) -> str:
    """
    Converts a function or class's structured data into a single text block
    that captures its meaning well enough for embedding.
    """
    name = node_data.get("name", "")
    docstring = node_data.get("docstring", "")
    node_type = node_data.get("type", "")

    text = f"{node_type} {name}. {docstring}"
    return text.strip()


def index_repo(graph, repo_name: str):
    """
    Walks every function/class node in the graph, embeds it,
    and upserts it into Pinecone under this repo's namespace.
    """
    ensure_index_exists()
    index = pc.Index(INDEX_NAME)

    vectors_to_upsert = []

    for node_id, node_data in graph.nodes(data=True):
        # Only index functions and classes - files are too broad to embed usefully
        if node_data.get("type") not in ("function", "class"):
            continue

        text = build_searchable_text(node_id, node_data)
        if not text.strip():
            continue

        embedding = embedding_model.encode(text).tolist()

        vectors_to_upsert.append({
            "id": node_id,
            "values": embedding,
            "metadata": {
                "name": node_data.get("name", ""),
                "type": node_data.get("type", ""),
                "file_path": node_data.get("file_path", ""),
                "docstring": node_data.get("docstring", "")[:500]  # keep metadata small
            }
        })

    # Pinecone recommends batching upserts - we do 100 at a time
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=repo_name)

    return len(vectors_to_upsert)


def search_repo(query: str, repo_name: str, top_k: int = 5) -> list:
    """
    Embeds the user's question and searches Pinecone for the most
    semantically similar functions/classes within this repo's namespace.
    """
    index = pc.Index(INDEX_NAME)

    query_embedding = embedding_model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=repo_name,
        include_metadata=True
    )

    matches = []
    for match in results["matches"]:
        matches.append({
            "node_id": match["id"],
            "score": match["score"],
            "name": match["metadata"].get("name", ""),
            "type": match["metadata"].get("type", ""),
            "file_path": match["metadata"].get("file_path", ""),
            "docstring": match["metadata"].get("docstring", "")
        })

    return matches