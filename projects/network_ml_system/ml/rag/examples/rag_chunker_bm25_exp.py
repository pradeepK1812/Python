"""
Module: rag_chunker_demo.py

Demonstrates the complete Retrieval-Augmented Generation (RAG)
pipeline using the framework.
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
from ml.rag.retrievers.bm25_retriever import BM25Retriever
from ml.rag.domain import RetrievedContext
from ml.rag.domain import GeneratedAnswer, Metadata, RetrievedContext
from ml.rag.generators.default_generator import DefaultGenerator
from ml.rag.llms.base import LLM
from ml.rag.chunker import chunk
from ml.rag.parser import parse
from ml.rag.examples.rag_chunker_exp import chunk_by_paragraph







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

       #Check retrievel via BM25 retriever

    query = "Why does HTTP use TCP?"

    retriever = BM25Retriever(
      chunks=chunks,
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
        print("Score:", context.score)
        print(context.content)

        print()

        print("Metadata")
        print(context.metadata)
        print("-" * 60)
"""
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
"""
