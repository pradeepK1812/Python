from ml.rag.domain import Chunk, Document, EmbeddedChunk, Section
from ml.rag.vector_stores.faiss_hnsw_vector_store import FAISSHNSWVectorStore


dimension = 3

store = FAISSHNSWVectorStore(
    dimension=dimension,
    M=2,
    ef_search=10,
)


document = Document(
    name="test.txt",
    path="/tmp/test.txt",
    content="test document",
)

section = Section(
    title="Test",
    level=1,
    content="test content",
)


chunks = [
    EmbeddedChunk(
        chunk=Chunk(
            chunk_id="chunk-1",
            source_document=document,
            section=section,
            chunk_index=0,
            content="first chunk",
        ),
        embedding=[1.0, 0.0, 0.0],
        embedding_model="test-model",
    ),
    EmbeddedChunk(
        chunk=Chunk(
            chunk_id="chunk-2",
            source_document=document,
            section=section,
            chunk_index=1,
            content="second chunk",
        ),
        embedding=[0.0, 1.0, 0.0],
        embedding_model="test-model",
    ),
    EmbeddedChunk(
        chunk=Chunk(
            chunk_id="chunk-3",
            source_document=document,
            section=section,
            chunk_index=2,
            content="third chunk",
        ),
        embedding=[1.0, 1.0, 0.0],
        embedding_model="test-model",
    ),

    EmbeddedChunk(
        chunk=Chunk(
            chunk_id="chunk-4",
            source_document=document,
            section=section,
            chunk_index=2,
            content="fourth  chunk",
        ),
        embedding=[1.0, 1.0, 1.0],
        embedding_model="test-model",
    ),

]

store.add(chunks)
print("Vectors indexed:", store._index.ntotal)
print("Chunks stored:", len(store._chunks))
print("Chunk IDs:", [chunk.chunk.chunk_id for chunk in store._chunks])

#serch for one query and print serach result
query = [1.0, 0.0,0.0]

results = store.search(
    query,
    top_k=3,
)

print("Search results:")

for result in results:
    print(
        result.chunk.chunk_id,
        result.chunk.content,
    )
