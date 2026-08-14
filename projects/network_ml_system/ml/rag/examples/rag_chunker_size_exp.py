"""
Module: rag_chunker_size_exp.py

Demonstrates the complete Retrieval-Augmented Generation (RAG)
pipeline using the framework.

Overview
--------
This example builds a small in-memory knowledge base, stores it in
the configured vector store, retrieves the most relevant contextual
knowledge for a query, and generates an answer using the configured
LLM.

Purpose
-------
- Demonstrate end-to-end RAG.
-Addition of real chunker instead of the hardcoded chunks
- Validate integration of the framework abstractions.

"""
#LLM import
from ml.rag.llms.ollama import OllamaLLM

from ml.rag.domain import (
    Document,
    StructuredDocument,
    Section,
    Chunk,
    EmbeddedChunk,
    VectorStore,
)
from ml.rag.reader import read_documents
from ml.rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from ml.rag.retrievers.retriever import Retriever
from ml.rag.retrievers.vector_store_retriever import VectorStoreRetriever
from ml.rag.domain import RetrievedContext
from ml.rag.vector_stores.chroma_vector_store import ChromaVectorStore
from ml.rag.domain import GeneratedAnswer, Metadata, RetrievedContext
from ml.rag.generators.default_generator import DefaultGenerator
from ml.rag.llms.base import LLM
from ml.rag.chunker import chunk
from ml.rag.parser import parse


#define the fixed chunk size for experiment
CHUNK_SIZE = 230
# creates chunk from the paragraph created from the Section
def _create_experimental_chunk(
    document: StructuredDocument,
    section: Section,
    chunk_index: int,
    content: str,
) -> Chunk:

    return Chunk(
        chunk_id=f"chunk_{chunk_index:04d}",
        source_document=document.source_document,
        section=section,
        chunk_index=chunk_index,
        content=content,
    )




#creates chunk by paragraph boundaries

def chunk_by_fixed_size(
    document: StructuredDocument,
    chunk_size: int,
) -> list[Chunk]:

    chunks = []
    chunk_index = 0

    for section in document.sections:

        content = section.content.strip()

        for start in range(0, len(content), chunk_size):

            chunk_content = content[
                start:start + chunk_size
            ]

            chunks.append(
                _create_experimental_chunk(
                    document,
                    section,
                    chunk_index,
                    chunk_content,
                )
            )

            chunk_index += 1

    return chunks











if __name__ == "__main__":

    documents = read_documents(
        "ml/rag/examples/chunker_exp_docs",
    )

    print(f"Loaded {len(documents)} document(s)")

    for document in documents:
        print("-" * 60)
        print(document.name)
        print(document.path)
        print(f"{len(document.content)} characters")

    document = documents[0]

    #parse the document to create the structured document
    strDoc = parse(document)  

    section = strDoc.sections
    chunks = chunk_by_fixed_size(strDoc,CHUNK_SIZE)

    print()
    print(f"Created {len(chunks)} chunk(s)")

    for chunk in chunks:
        print("-" * 60)
        print(chunk.chunk_id)
        print(f"Characters: {len(chunk.content)}")
        print(f"Words: {len(chunk.content.split())}")
        print(chunk.content)

    embedding_model = SentenceTransformerEmbeddingModel(
      model_name="all-MiniLM-L6-v2",
    )
    
    embedded_chunks = []

    for chunk in chunks:

        vector = embedding_model.embed(
            chunk.content,
        )

        embedded_chunks.append(
            EmbeddedChunk(
                chunk=chunk,
                embedding=vector,
                embedding_model=embedding_model.model_name,
            )
        )
    print(f"Embedding Model: {embedding_model.model_name}")

    #embedded_chunks = embedding_model.embed(chunks,)

    print(f"Generated {len(embedded_chunks)} embeddings")

    for embedded_chunk in embedded_chunks:
        print("-" * 60)
        print(embedded_chunk.chunk.chunk_id)
        print(embedded_chunk.embedding_model)
        print(len(embedded_chunk.embedding))
        print(f"Vector dimension : {len(embedded_chunk.embedding)}")

    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="rag_chunker_exp",
    )
    vector_store.add(embedded_chunks,)
    print()
    print("Stored embeddings in Chroma.")
    print(f"Collection : {vector_store._collection_name}")
    print(f"Chunks      : {len(embedded_chunks)}")
    
    
