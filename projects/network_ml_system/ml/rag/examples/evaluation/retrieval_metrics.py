
# test metrices definitions


def recall_at_k(
    retrieved_chunks: list[str],
    relevant_chunks: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K for a ranked list of retrieved chunk IDs.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")

    if not relevant_chunks:
        raise ValueError("relevant_chunks must not be empty.")

    if not retrieved_chunks:
        return 0.0

    k = min(k, len(retrieved_chunks))

    retrieved_at_k = retrieved_chunks[:k]

    relevant_retrieved = sum(
        1
        for chunk_id in retrieved_at_k
        if chunk_id in relevant_chunks
    )

    return relevant_retrieved / len(relevant_chunks)




def precision_at_k(
    retrieved_chunks: list[str],
    relevant_chunks: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K for a ranked list of retrieved chunk IDs.
    """

    if k < 1:
        raise ValueError("k must be greater than zero.")

    if not retrieved_chunks:
        return 0.0

    k = min(k, len(retrieved_chunks))

    retrieved_at_k = retrieved_chunks[:k]

    relevant_retrieved = sum(
        1
        for chunk_id in retrieved_at_k
        if chunk_id in relevant_chunks
    )

    return relevant_retrieved / k




