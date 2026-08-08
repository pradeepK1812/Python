"""
Module: rag_chunker_demo.py

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

#Splits Section  into paragraph

def _split_into_paragraphs(content: str) -> list[str]:
    paragraphs = content.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


#creates chunk by paragraph boundaries

def chunk_by_paragraph(
    document: StructuredDocument,
) -> list[Chunk]:

    chunks = []
    chunk_index = 0

    for section in document.sections:

        paragraphs = _split_into_paragraphs(
            section.content
        )

        for paragraph in paragraphs:

            chunks.append(
                _create_experimental_chunk(
                    document,
                    section,
                    chunk_index,
                    paragraph,
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
    chunks = chunk_by_paragraph(strDoc)

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
        collection_name="rag_chunker_exp",
    )
    vector_store.add(embedded_chunks,)

    print()
    print("Stored embeddings in Chroma.")
    print(f"Collection : {vector_store._collection_name}")
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

    llm = OllamaLLM(
        model_name="llama3.2:1b",
    )

    generator = DefaultGenerator(
         llm=llm,
    )

    print()
    print("Generator")
    print("---------")
    print(type(generator).__name__)

    print("LLM")
    print("---")
    print(llm.model_name)

    print()
    print("Generating answer...")
    print("-" * 60)

    generated_answer = generator.generate(
        query=query,
        retrieved_contexts=retrieved_contexts,
    )

    print()
    print("Generated Answer")
    print("----------------")
    print(generated_answer.answer)
