import pytest

from ml.rag.domain import (
    Document,
    Section,
    Chunk,
    EmbeddedChunk,
)

from ml.rag.vector_stores.chroma_vector_store import ChromaVectorStore

@pytest.fixture

def sample_document():
    return Document(
        name="sample.md",
        path="/docs/sample.md",
        content="# Sample",
    )


@pytest.fixture

def sample_section():
    return Section(
        title="Introduction",
        level=1,
        content="Hello",
    )

@pytest.fixture

def sample_chunk(sample_document, sample_section):
    return Chunk(
        chunk_id="chunk-1",
        source_document=sample_document,
        section=sample_section,
        chunk_index=0,
        content="Hello World",
    ) 

@pytest.fixture

def sample_embedded_chunk(sample_chunk):
    return EmbeddedChunk(
        chunk=sample_chunk,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="all-MiniLM-L6-v2",
    )

@pytest.fixture
def sample_chroma_record(sample_embedded_chunk):
    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test",
    )
    return vector_store._to_chroma(sample_embedded_chunk)
