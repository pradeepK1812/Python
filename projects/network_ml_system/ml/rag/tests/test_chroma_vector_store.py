

from ml.rag.domain import Document

from ml.rag.domain import EmbeddedChunk, VectorStore,Section,Chunk

def test_import():
    document = Document(
        name="sample.md",
        path="/tmp/sample.md",
        content="Hello",
    )

    assert document.name == "sample.md"


from ml.rag.vector_stores.chroma_vector_store import ChromaVectorStore


def test_build_metadata(sample_embedded_chunk):

    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test",
    )

    metadata = vector_store._build_metadata(sample_embedded_chunk)

    assert metadata == {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_index": 0,
        "document_name": "sample.md",
        "document_path": "/docs/sample.md",
        "section_title": "Introduction",
        "section_level": 1,
    }



from ml.rag.vector_stores.chroma_vector_store import _ChromaRecord


def test_to_chroma(
    sample_embedded_chunk
):
    """Verify that an EmbeddedChunk is correctly converted to a Chroma record."""
    
    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test",
    )
    # Arrange
    chunk = sample_embedded_chunk.chunk

    # Act
    record = vector_store._to_chroma(sample_embedded_chunk)

    # Assert
    assert isinstance(record, _ChromaRecord)

    assert record.id == chunk.chunk_id
    assert record.document == chunk.content
    assert record.embedding == sample_embedded_chunk.embedding

    assert record.metadata == {
        "embedding_model": sample_embedded_chunk.embedding_model,
        "chunk_index": chunk.chunk_index,
        "document_name": chunk.source_document.name,
        "document_path": chunk.source_document.path,
        "section_title": chunk.section.title,
        "section_level": chunk.section.level,
    }



def test_to_chroma_records(sample_embedded_chunk):
    """Verify that a list of EmbeddedChunks is converted into Chroma records."""

    # Arrange
    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test",
    )

    chunks = [
        sample_embedded_chunk,
        sample_embedded_chunk,
    ]

    # Act
    records = vector_store._to_chroma_records(chunks)

    # Assert
    assert isinstance(records, list)
    assert len(records) == 2

    assert all(isinstance(record, _ChromaRecord) for record in records)

    assert records[0] == vector_store._to_chroma(sample_embedded_chunk)
    assert records[1] == vector_store._to_chroma(sample_embedded_chunk)


def test_from_chroma(sample_embedded_chunk,sample_chroma_record):
   
    # Arrange
    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test",
    )

    expected = sample_embedded_chunk

    #Act
    embedded_chunk = vector_store._from_chroma(sample_chroma_record)

    #Assert
    assert isinstance(embedded_chunk, EmbeddedChunk)
    assert embedded_chunk.embedding_model == expected.embedding_model
    assert embedded_chunk.embedding == expected.embedding
    assert embedded_chunk.chunk.chunk_id == expected.chunk.chunk_id
    assert embedded_chunk.chunk.content == expected.chunk.content
    assert embedded_chunk.chunk.source_document.name == expected.chunk.source_document.name
    assert embedded_chunk.chunk.chunk_index == expected.chunk.chunk_index

    assert embedded_chunk.chunk.source_document.path == (
           expected.chunk.source_document.path
    )

    assert embedded_chunk.chunk.section.level == (
           expected.chunk.section.level
    )
    assert embedded_chunk.chunk.section.title == expected.chunk.section.title


def test_embedded_chunk_roundtrip_conversion(sample_embedded_chunk):
        """
        Verify that an EmbeddedChunk survives a full
        EmbeddedChunk -> ChromaRecord -> EmbeddedChunk round trip.
        """

        # Arrange
        vector_store = ChromaVectorStore(
            persist_directory=":memory:",
            collection_name="test",
        )
        expected = sample_embedded_chunk

        # Act
        chroma_record = vector_store._to_chroma(expected)
        recovered_chunk = vector_store._from_chroma(chroma_record)

        # Assert
        assert isinstance(recovered_chunk, EmbeddedChunk)

        assert recovered_chunk.embedding == expected.embedding
        assert recovered_chunk.embedding_model == expected.embedding_model

        assert recovered_chunk.chunk.chunk_id == expected.chunk.chunk_id
        assert recovered_chunk.chunk.chunk_index == expected.chunk.chunk_index
        assert recovered_chunk.chunk.content == expected.chunk.content

        assert recovered_chunk.chunk.source_document.name == (
            expected.chunk.source_document.name
        )
        assert recovered_chunk.chunk.source_document.path == (
            expected.chunk.source_document.path
        )

        assert recovered_chunk.chunk.section.title == (
            expected.chunk.section.title
        )
        assert recovered_chunk.chunk.section.level == (
            expected.chunk.section.level
        )
               # 6. Assert Section metadata matches
        assert recovered_chunk.chunk.section.title == sample_embedded_chunk.chunk.section.title
        assert recovered_chunk.chunk.section.level == sample_embedded_chunk.chunk.section.level



def test_add(sample_embedded_chunk):
    # Arrange
    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test",
    )
    expected = sample_embedded_chunk

    # Act
    vector_store.add([expected])

    assert vector_store._collection.count() == 1

    stored_data = vector_store._collection.get()

    # Assert stored document
    assert stored_data["ids"] == [expected.chunk.chunk_id]
    assert stored_data["documents"] == [expected.chunk.content]

    # Assert metadata
    metadata = stored_data["metadatas"][0]

    assert metadata["chunk_index"] == expected.chunk.chunk_index
    assert metadata["embedding_model"] == expected.embedding_model
    assert metadata["document_name"] == expected.chunk.source_document.name
    assert metadata["document_path"] == expected.chunk.source_document.path
    assert metadata["section_title"] == expected.chunk.section.title
    assert metadata["section_level"] == expected.chunk.section.level




def test_search(sample_embedded_chunk):
    # Arrange
    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test_search_collection",
    )

    # Base domain objects
    doc = sample_embedded_chunk.chunk.source_document
    section = sample_embedded_chunk.chunk.section
    model = sample_embedded_chunk.embedding_model

    # Chunk A: matches vector [1.0, 0.0, 0.0]
    chunk_a = EmbeddedChunk(
        chunk=Chunk(
            chunk_id="chunk-a",
            source_document=doc,
            section=section,
            chunk_index=0,
            content="Content for Chunk A",
        ),
        embedding=[1.0, 0.0, 0.0],
        embedding_model=model,
    )

    # Chunk B: distinct vector [0.0, 1.0, 0.0]
    chunk_b = EmbeddedChunk(
        chunk=Chunk(
            chunk_id="chunk-b",
            source_document=doc,
            section=section,
            chunk_index=1,
            content="Content for Chunk B",
        ),
        embedding=[0.0, 1.0, 0.0],
        embedding_model=model,
    )

    # Chunk C: distinct vector [0.0, 0.0, 1.0]
    chunk_c = EmbeddedChunk(
        chunk=Chunk(
            chunk_id="chunk-c",
            source_document=doc,
            section=section,
            chunk_index=2,
            content="Content for Chunk C",
        ),
        embedding=[0.0, 0.0, 1.0],
        embedding_model=model,
    )

    # Act
    vector_store.add([chunk_a, chunk_b, chunk_c])

    # Search with embedding exact-matching Chunk A, asking for top 1 result
    results = vector_store.search(embedding=[1.0, 0.0, 0.0], top_k=1)

    # Assert
    returned_chunk = results[0]
    assert isinstance(returned_chunk, EmbeddedChunk)
    
    # 1. Exactly one result returned
    assert len(results) == 1

    # 2. Correct chunk (Chunk A) returned
    expected = chunk_a
    assert returned_chunk.chunk.chunk_id == expected.chunk.chunk_id
    assert returned_chunk.chunk.content == expected.chunk.content
    assert list(returned_chunk.embedding) == list(expected.embedding)
    assert returned_chunk.embedding_model == expected.embedding_model
