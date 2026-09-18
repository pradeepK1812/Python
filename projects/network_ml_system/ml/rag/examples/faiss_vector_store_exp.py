from ml.rag.domain import Chunk, Document, Section, EmbeddedChunk
from ml.rag.vector_stores.faiss_vector_store import FAISSVectorStore


document = Document(
    name="test",
    path="test.txt",
    content="",
)

section = Section(
    title="Test",
    level=1,
    content="",
)


def make_chunk(chunk_id, embedding):
    chunk = Chunk(
        chunk_id=chunk_id,
        source_document=document,
        section=section,
        chunk_index=0,
        content=chunk_id,
    )

    return EmbeddedChunk(
        chunk=chunk,
        embedding=embedding,
        embedding_model="test-model",
    )


chunks = [
    make_chunk("chunk_0", [1, 0, 0]),
    make_chunk("chunk_1", [0, 1, 0]),
    make_chunk("chunk_2", [1, 1, 0]),
]


store = FAISSVectorStore(
    dimension=3,
)

store.add(chunks)

print("FAISS vectors:", store._index.ntotal)

results = store.search(
    embedding=[1, 0, 0],
    top_k= 10,
)

print("Search results:")

for result in results:
    print(
        result.chunk.chunk_id,
        result.embedding,
    )
