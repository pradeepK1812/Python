import faiss
import numpy as np


dimension = 128 #dimension of the vector
M = 2 #No of connections a node can make 

index = faiss.IndexHNSWFlat(
    dimension,
    M,
    faiss.METRIC_INNER_PRODUCT,
)

rng = np.random.default_rng(42)

raw_vectors = rng.standard_normal((1000, dimension))

norms = np.linalg.norm(
    raw_vectors,
    axis=1,
    keepdims=True,
)

unit_vectors = raw_vectors / norms

#Ensure array is 2D, contiguous, and float32
unit_vectors = np.ascontiguousarray(unit_vectors, dtype=np.float32)

faiss.normalize_L2(unit_vectors)

index.add(unit_vectors)

print("Vectors indexed:", index.ntotal)

query = np.array(
    [[1, 0, 0]],
    dtype=np.float32,
)
# Pad 0 values at the front and 125 values at the end along the 2nd axis (columns)
query_128 = np.pad(
    query, 
    pad_width=((0, 0), (0, 125)), 
    mode='constant', 
    constant_values=0
)

# Ensure contiguous memory layout for libraries like FAISS
query_128 = np.ascontiguousarray(query_128, dtype=np.float32)

faiss.normalize_L2(query_128)

index.hnsw.efSearch = 500
print("dimension:", dimension)
print("efSearch:", index.hnsw.efSearch)

distances, indices = index.search(
    query_128,
    4,
)

print("Distances:", distances)
print("Indices:", indices)

exact_index = faiss.IndexFlatIP(dimension)

exact_index.add(unit_vectors)

exact_distances, exact_indices = exact_index.search(
    query_128,
    4,
)

print("Exact distances:", exact_distances)
print("Exact indices:", exact_indices)
