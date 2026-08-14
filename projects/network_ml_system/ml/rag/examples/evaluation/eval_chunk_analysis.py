

from ml.rag.vector_stores.chroma_vector_store import (
    ChromaVectorStore,
)
from ml.rag.retrievers.vector_store_retriever import (
    VectorStoreRetriever,
)


section_vector_store = ChromaVectorStore(
    persist_directory=":memory:",
    collection_name="rag_demo",
)

chunker_vector_store = ChromaVectorStore(
    persist_directory=":memory:",
    collection_name="rag_chunker_exp",
)

"""
print(
    "rag_demo count:",
    section_vector_store._collection.count(),
)

print(
    "rag_chunker_exp count:",
    chunker_vector_store._collection.count(),
)
"""


print("=" * 60)
print("rag_demo")
print("=" * 60)

section_results = section_vector_store._collection.get(
    include=["documents", "metadatas"],
)

for chunk_id, document, metadata in zip(
    section_results["ids"],
    section_results["documents"],
    section_results["metadatas"],
):
    character_count = len(document)
    word_count = len(document.split())
    print("-" * 60)
    print(f"Chunk ID : {chunk_id}")
    print(f"Characters : {character_count}")
    print(f"Words : {word_count}")
    #print(f"Metadata : {metadata}")
    print("Content:")
    print(document)


print()
print("=" * 60)
print("rag_chunker_exp")
print("=" * 60)

chunker_results = chunker_vector_store._collection.get(
    include=["documents", "metadatas"],
)

for chunk_id, document, metadata in zip(
    chunker_results["ids"],
    chunker_results["documents"],
    chunker_results["metadatas"],
):
    character_count = len(document)
    word_count = len(document.split())
    print("-" * 60)
    print(f"Chunk ID : {chunk_id}")
    print(f"Characters : {character_count}")
    print(f"Words : {word_count}")
    #print(f"Metadata : {metadata}")
    print("Content:")
    print(document)
