from ml.rag.examples.evaluation.retrieval_metrics import precision_at_k
from ml.rag.examples.evaluation.retrieval_metrics import recall_at_k 
from ml.rag.examples.evaluation.retrieval_metrics import reciprocal_rank,mean_reciprocal_rank 

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

def test_reciprocal_rank_first_result_relevant():
    retrieved = [
        "chunk_0002",
        "chunk_0001",
        "chunk_0003",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 1.0


def test_reciprocal_rank_second_result_relevant():
    retrieved = [
        "chunk_0001",
        "chunk_0002",
        "chunk_0003",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 0.5


def test_reciprocal_rank_third_result_relevant():
    retrieved = [
        "chunk_0001",
        "chunk_0003",
        "chunk_0004",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 1 / 3


def test_reciprocal_rank_no_relevant_result():
    retrieved = [
        "chunk_0001",
        "chunk_0003",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 0.0


def test_reciprocal_rank_uses_first_relevant_result():
    retrieved = [
        "chunk_0001",
        "chunk_0003",
        "chunk_0002",
        "chunk_0000",
    ]

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    # First relevant result is at rank 3.
    # chunk_0000 at rank 4 must not change the result.
    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 1 / 3


def test_reciprocal_rank_empty_retrieved():
    retrieved = []

    relevant = {
        "chunk_0002",
        "chunk_0000",
        "chunk_0004",
    }

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == 0.0


def test_reciprocal_rank_empty_relevant():
    retrieved = [
        "chunk_0002",
    ]

    relevant = set()

    try:
        reciprocal_rank(
            retrieved,
            relevant,
        )
        assert False
    except ValueError:
        pass


def test_mean_reciprocal_rank():
    reciprocal_ranks = [
        1.0,
        0.5,
        0.25,
    ]

    assert mean_reciprocal_rank(
        reciprocal_ranks,
    ) == (1.0 + 0.5 + 0.25) / 3


def test_mean_reciprocal_rank_perfect():
    reciprocal_ranks = [
        1.0,
        1.0,
        1.0,
    ]

    assert mean_reciprocal_rank(
        reciprocal_ranks,
    ) == 1.0


def test_mean_reciprocal_rank_no_relevant_results():
    reciprocal_ranks = [
        0.0,
        0.0,
        0.0,
    ]

    assert mean_reciprocal_rank(
        reciprocal_ranks,
    ) == 0.0



def test_mean_reciprocal_rank_empty_input():
    reciprocal_ranks = []

    try:
        mean_reciprocal_rank(
            reciprocal_ranks,
        )
        assert False
    except ValueError:
        pass



