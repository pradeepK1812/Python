"""
Module: rag_demo.py

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
- Validate integration of the framework abstractions.
- Serve as the "Hello World" example for the framework.

"""
from ml.rag.domain import (
    Document,
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


if __name__ == "__main__":

    documents = read_documents(
        "ml/rag/examples/",
    )

    print(f"Loaded {len(documents)} document(s)")

    for document in documents:
        print("-" * 60)
        print(document.name)
        print(document.path)
        print(f"{len(document.content)} characters")

    document = documents[0]

    section = Section(
        title="Networking Basics",
        level=1,
        content="",
    )

    chunks = [
                Chunk(
                    chunk_id="tcp-001",
                    source_document=document,
                    section=section,
                    chunk_index=0,
                    content=(
                        "TCP is a connection-oriented transport protocol. "
                        "It provides reliable and ordered delivery of data."
                    ),
                ),

                Chunk(
                            chunk_id="udp-001",
                            source_document=document,
                            section=section,
                            chunk_index=1,
                            content=(
                                        " UDP (User Datagram Protocol) is a connectionless transport protocol. "
                                          "It does not guarantee delivery, ordering, or duplicate protection. "
                                    ),
                     ),

                Chunk(
                            chunk_id="http-001",
                            source_document=document,
                            section=section,
                            chunk_index=2,
                            content=(
                                     "HTTP (Hypertext Transfer Protocol) is an application layer protocol\
                                      used for communication between web clients and web servers. "
                                      "HTTP typically runs over TCP because reliable and ordered delivery is\
                                      required for web content. "

                                    ),
                     ),
                
                Chunk(
                            chunk_id="dns-001",
                            source_document=document,
                            section=section,
                            chunk_index=3,
                            content=(
                                       "DNS (Domain Name System) translates domain names into IP addresses. "
                                       "Most DNS queries use UDP because the request and response messages\
                                       are typically small and low latency is preferred. TCP may be used\
                                       for large DNS responses or zone transfers."



                                    ),
                     ),
             ]

    print()
    print(f"Created {len(chunks)} chunk(s)")

    for chunk in chunks:
        print("-" * 60)
        print(chunk.chunk_id)
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
        #print(len(embedded_chunk.embedding))
        print(f"Vector dimension : {len(embedded_chunk.embedding)}")

    vector_store = ChromaVectorStore(
        persist_directory=":memory:",
        collection_name="test",
    )
    vector_store.add(embedded_chunks,)

    print()
    print("Stored embeddings in Chroma.")
    print(f"Collection : rag_demo")
    print(f"Chunks      : {len(embedded_chunks)}")
    
    #Check retrievel from Chroma DB

    query = "Why does HTTP use TCP?"

    query_embedding = embedding_model.embed(
        query,
    )

    results = vector_store.search(
        query_embedding,
        top_k=2,
    )

    print()
    print(f"Query: {query}")
    print()

    for result in results:
        print("-" * 60)
        print(result.chunk.chunk_id)
        print(result.chunk.content)

    retriever = VectorStoreRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    retrieved_contexts = retriever.retrieve(
        query=query,
        top_k=2,
    )

    print()
    print("Retrieved Contexts")
    print("--------------------")

    for context in retrieved_contexts:
        print("-" * 60)
        print(context.content)

        print()

        print("Metadata")
        print(context.metadata)
