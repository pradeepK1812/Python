from ml.rag.examples.evaluation.retrieval_metrics import precision_at_k
from ml.rag.examples.evaluation.retrieval_metrics import recall_at_k 

def test_precision_at_k_all_relevant():
    retrieved = [
        "chunk_0002",
        "chunk_0000",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert precision_at_k(
        retrieved,
        relevant,
        2,
    ) == 1.0


def test_precision_at_k_partial_relevance():
    retrieved = [
        "chunk_0002",
        "chunk_0003",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert precision_at_k(
        retrieved,
        relevant,
        2,
    ) == 0.5


def test_precision_at_k_no_relevant_results():
    retrieved = [
        "chunk_0001",
        "chunk_0003",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert precision_at_k(
        retrieved,
        relevant,
        2,
    ) == 0.0


def test_precision_at_k_invalid_k():
    retrieved = ["chunk_0002"]

    relevant = {"chunk_0002"}

    try:
        precision_at_k(retrieved, relevant, 0)
        assert False
    except ValueError:
        pass


def test_recall_at_k():
    retrieved = [
        "chunk_0002",
        "chunk_0000",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert recall_at_k(
        retrieved,
        relevant,
        2,
    ) == 2 / 3


def test_recall_at_k_full_recall():
    retrieved = [
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert recall_at_k(
        retrieved,
        relevant,
        3,
    ) == 1.0
    


def test_recall_at_k_zero_recall():
    retrieved = [
        "chunk_0001",
        "chunk_0003",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert recall_at_k(
        retrieved,
        relevant,
        2,
    ) == 0.0




def test_recall_at_k_invalid_k():
    retrieved = ["chunk_0002"]

    relevant = {"chunk_0002"}

    try:
        recall_at_k(
            retrieved,
            relevant,
            0,
        )
        assert False
    except ValueError:
        pass


def test_recall_at_k_empty_relevant_chunks():
    retrieved = ["chunk_0002"]

    try:
        recall_at_k(
            retrieved,
            set(),
            1,
        )
        assert False
    except ValueError:
        pass


