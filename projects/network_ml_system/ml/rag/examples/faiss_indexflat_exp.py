import numpy as np
import faiss


vectors = np.array(
    [
        [1, 0],
        [0, 1],
        [1, 1],
        [5, 5],
    ],
    dtype=np.float32,
)

faiss.normalize_L2(vectors)
query = np.array(
    [
        [1, 0],
    ],
    dtype=np.float32,
)

faiss.normalize_L2(query)

index = faiss.IndexFlatIP(2)

index.add(vectors)
print("Number of vectors:", index.ntotal)

distances, indices = index.search(
    query,
    k=4,
)

print("Distances:", distances)
print("Indices:", indices)
