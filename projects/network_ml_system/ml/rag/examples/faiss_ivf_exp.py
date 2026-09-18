import faiss
import numpy as np

dimension = 3
nlist = 2

quantizer = faiss.IndexFlatIP(dimension)

index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_INNER_PRODUCT,
)

vectors = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)

faiss.normalize_L2(vectors)

print("Before training:", index.is_trained)

index.train(vectors)

print("After training:", index.is_trained)

index.add(vectors)

query = np.array(
    [[1, 0, 0]],
    dtype=np.float32,
)

faiss.normalize_L2(query)

index.nprobe = 2

distances, indices = index.search(
    query,
    4,
)

print("nprobe:", index.nprobe)
print("Distances:", distances)
print("Indices:", indices)
print("Vectors indexed:", index.ntotal)

