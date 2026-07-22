"""
Ingestion pipeline class definition and Api definition

"""

from pathlib import Path




    class IngestionPipeline:

        def __init__(
            self,
            reader: MarkdownReader,
            parser: MarkdownParser,
            chunker: Chunker,
            embedder: Embedder,
            vector_store: VectorStore,
        ) -> None:

            self.reader = reader
            self.parser = parser
            self.chunker = chunker
            self.embedder = embedder
            self.vector_store = vector_store

        def ingest_document(self, path: Path) -> None:
            
            """
            Take file for ingestion
            """

            document = self.reader.read(path)

            structured_document = self.parser.parse(document)

            chunks = self.chunker.chunk(structured_document)

            embedded_chunks = self.embedder.embed(chunks)

            self.vector_store.add(embedded_chunks)

        def ingest_directory(self, directory: Path):
            """
            Takes directory path for ingestion
            """
            
            root = Path(directory)
            if not root.exists():
               raise FileNotFoundError(f"Root directory does not exist: {root.resolve()}")
            if not root.is_dir():
               raise NotADirectoryError(
                f"'{root}' is not a valid directory."
            )
            for md_file in root.rglob("*.md"):
                
                try:
                   self.ingest_document(md_file)
                except Exception as ex:
                    raise RuntimeError(
                       f"Failed to ingest '{md_file}'."
                    ) from ex
               
