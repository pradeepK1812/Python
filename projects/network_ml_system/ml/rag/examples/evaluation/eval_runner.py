import sys
import json
from ml.rag.retrievers.retriever import Retriever
from ml.rag.examples.evaluation.retrieval_metrics import precision_at_k,recall_at_k
from ml.rag.examples.evaluation.retrieval_metrics import reciprocal_rank,mean_reciprocal_rank 
from ml.rag.examples.evaluation.evaluation_utils import chunks_to_knowledge 

from ml.rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from ml.rag.vector_stores.chroma_vector_store import (
    ChromaVectorStore,
)
from ml.rag.retrievers.vector_store_retriever import (
    VectorStoreRetriever,
)


# Path to eval_truth.json file
truth_json_file_path = "ml/rag/examples/evaluation/eval_truth.json"
knowledge_json_file_path = "ml/rag/examples/evaluation/chunk_knowledge_map.json"

embedding_model = SentenceTransformerEmbeddingModel(
    model_name="all-MiniLM-L6-v2",
)

section_vector_store = ChromaVectorStore(
    persist_directory=":memory:",
    collection_name="rag_demo",
)

chunker_vector_store = ChromaVectorStore(
    persist_directory=":memory:",
    collection_name="rag_chunker_exp",
)

rebalanced_vector_store = ChromaVectorStore(
    persist_directory=":memory:",
    collection_name="rag_chunker_rebalance_exp",
)

section_retriever = VectorStoreRetriever(
    embedding_model=embedding_model,
    vector_store=section_vector_store,
)

chunker_retriever = VectorStoreRetriever(
    embedding_model=embedding_model,
    vector_store=chunker_vector_store,
)

rebalanced_retriever = VectorStoreRetriever(
    embedding_model=embedding_model,
    vector_store=rebalanced_vector_store,
)

retrievers = [
    ("rag_demo", section_retriever),
    ("rag_chunker_exp", chunker_retriever),
    ("rag_chunker_rebalance_exp", rebalanced_retriever),
]

print(
    "rag_demo count:",
    section_vector_store._collection.count(),
)

print(
    "rag_chunker_exp count:",
    chunker_vector_store._collection.count(),
)

print(
    "rag_chunker_rebalance_exp count:",
    rebalanced_vector_store._collection.count(),
)


#=============================================
#added to create exit checkpoint for testing
#sys.exit(0)
#=============================================

#set K as 2 for evaluation
K =2
# 1. Load the JSON file into a list
with open(truth_json_file_path, "r") as f:
    evaluation_cases = json.load(f)


with open(knowledge_json_file_path, "r") as f:
    chunk_knowledge_map = json.load(f)

#list to store the reciprocal_rank, precisions and recalls
for name, retriever in retrievers:
    reciprocal_ranks = []
    precisions = []
    recalls = []
    knowledge_map = chunk_knowledge_map[name]
    for evaluation_case in evaluation_cases:

        query = evaluation_case["query"]

        retrieved_contexts = retriever.retrieve(
            query=query,
            top_k=K,
        )

        retrieved_chunks = [
            context.metadata["chunk_id"]
            for context in retrieved_contexts
        ]

        retrieved_knowledge = chunks_to_knowledge(
          retrieved_chunks,
          knowledge_map,
        )
         
        relevant_knowledge = set( evaluation_case["relevant_knowledge"])

        precision = precision_at_k(
            retrieved_knowledge,
            relevant_knowledge,
            K,
        )

        precisions.append(precision)

        recall = recall_at_k(
            retrieved_knowledge,
            relevant_knowledge,
            K,
        )

        recalls.append(recall)

        rr = reciprocal_rank(
            retrieved_knowledge,
            relevant_knowledge,
        )

        reciprocal_ranks.append(rr)

        print("-" * 60)
        print(f"Query: {query}")
        print(f"Relevant : {sorted(relevant_knowledge)}")
        print(f"Retrieved: {retrieved_chunks}")
        print(f"Precision@{K}: {precision:.3f}")
        print(f"Recall@{K}   : {recall:.3f}")
        print(f"RR          : {rr:.3f}")

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mrr = mean_reciprocal_rank(reciprocal_ranks)

    print(f"Mean report for retriever:{name}")
    print()
    print("=" * 60)
    print("RAG Retrieval Evaluation")
    print("=" * 60)
    print(f"Queries          : {len(evaluation_cases)}")
    print(f"K                : {K}")
    print(f"Mean Precision@{K}: {mean_precision:.3f}")
    print(f"Mean Recall@{K}   : {mean_recall:.3f}")
    print(f"MRR              : {mrr:.3f}")
    print("=" * 60)
