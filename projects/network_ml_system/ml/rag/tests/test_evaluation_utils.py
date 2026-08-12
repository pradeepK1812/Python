
from ml.rag.examples.evaluation.evaluation_utils import chunks_to_knowledge


def test_chunks_to_knowledge():
    
    retrieved_chunks = ["chunk_0001", "chunk_0000"]

    knowledge_map = {
        "chunk_0000": ["tcp_characteristics"],
        "chunk_0001": ["udp_characteristics"],
    }

    result = chunks_to_knowledge(
        retrieved_chunks,
        knowledge_map,
    )

    assert result == {
        "tcp_characteristics",
        "udp_characteristics",
    }
