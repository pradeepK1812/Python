from pathlib import Path
from typing import Any
from dataclasses import dataclass


import chromadb
from chromadb.api.models.Collection import Collection

from ml.rag.domain import Document
from ml.rag.domain import EmbeddedChunk, VectorStore,Section,Chunk,EmbeddedChunk,Metadata

@dataclass(frozen=True, slots=True)
class _ChromaRecord:
    id: str
    document: str
    embedding: list[float]
    metadata: Metadata

class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of the VectorStore interface.

    Responsible for translating between the domain model
    (EmbeddedChunk) and ChromaDB's storage format.
    """

    _collection_name: str
    _persist_directory: Path
    _client: chromadb.PersistentClient
    _collection: Collection

    def __init__(
        self,
        collection_name: str,
        persist_directory: Path,
    ) -> None:

        self._collection_name = collection_name
        self._persist_directory = persist_directory

        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory)
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )
    
    def add(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        
        if not chunks:
         return

        records = self._to_chroma_records(chunks)

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for record in records:
            ids.append(record.id)
            documents.append(record.document)
            embeddings.append(record.embedding)
            metadatas.append(record.metadata)

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[EmbeddedChunk]:

        """
        Queries ChromaDB for the top_k most similar chunks given an embedding.
        """
        
        if top_k < 1:
           raise ValueError("top_k must be greater than zero.")
        
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "embeddings", "metadatas"],
        )
        # Chroma query returns lists of lists per input embedding query
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        chunks: list[EmbeddedChunk] = []
        for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            record = _ChromaRecord(
                id=id_,
                document=doc,
                embedding=emb,
                metadata=meta or {},
            )
            chunks.append(self._from_chroma(record))

        return chunks

    @staticmethod
    def _build_metadata(
        embedded_chunk: EmbeddedChunk
    ) -> dict[str, Any]:
       
        """
        Builds the metadata dictionary stored alongside each embedding in ChromaDB.
        """
        chunk = embedded_chunk.chunk

        return {
            "embedding_model": embedded_chunk.embedding_model,
            "chunk_index": chunk.chunk_index,
            "document_name": chunk.source_document.name,
            "document_path": str(chunk.source_document.path),
            "section_title": chunk.section.title,
            "section_level": chunk.section.level,
        }

            
    def _to_chroma(
        self,
        embedded_chunk: EmbeddedChunk,
    ) ->  _ChromaRecord:
        
        chunk = embedded_chunk.chunk

        return _ChromaRecord(
            id=chunk.chunk_id,
            document=chunk.content,
            embedding=embedded_chunk.embedding,
            metadata=self._build_metadata(embedded_chunk),
        )

           
    def _from_chroma(
      self,
      record: _ChromaRecord,
    ) -> EmbeddedChunk:

            metadata = record.metadata

            document = Document(
                name=metadata["document_name"],
                path=metadata["document_path"],
                content="",  # Full document content is intentionally not stored in the vector store.
            )

            section = Section(
                title=metadata["section_title"],
                level=metadata["section_level"],
                content="",  # Full section content is intentionally not stored in the vector store.
            )

            chunk = Chunk(
                chunk_id=record.id,
                source_document=document,
                section=section,
                chunk_index=metadata["chunk_index"],
                content=record.document,
            )

            return EmbeddedChunk(
                chunk=chunk,
                embedding=record.embedding,
                embedding_model=metadata["embedding_model"],
            )    
                
    def _to_chroma_records(
       self,
       chunks: list[EmbeddedChunk],
    ) -> list[_ChromaRecord]:

       """
       Converts a list of domain EmbeddedChunks to internal _ChromaRecords.
       """
       return [self._to_chroma(chunk) for chunk in chunks]

