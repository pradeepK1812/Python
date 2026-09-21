import faiss
import numpy as np
from ml.rag.vector_stores.faiss_ivf_vector_store import FAISSIVFVectorStore


from ml.rag.domain import (
    Document,
    Section,
    Chunk,
    EmbeddedChunk,
)

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

store = FAISSIVFVectorStore(
    dimension=3,
    nlist=2,
    nprobe=1,
)

vectors = [
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
]

chunks = [
    make_chunk(f"chunk_{i}", vector)
    for i, vector in enumerate(vectors)
]

store.train(vectors)
store.add(chunks)

query = np.array(
    [[1, 0, 0]],
    dtype=np.float32,
)

faiss.normalize_L2(query)

store.nprobe = 2

results = store.search(
    embedding=[1, 0, 0],
    top_k=4,
)
print("nprobe:", store.nprobe)
print("Result:", results)

