"""
Chunking Strategy (Version 1)
-----------------------------

Documents are split along semantic boundaries rather than
fixed token or character counts.

For Markdown documents, headings (#, ##, ###) define the
initial chunk boundaries.

This preserves semantic coherence and improves retrieval
quality.

Future versions may further subdivide large sections based
on retrieval performance.
"""

"""
Chunker Contract
----------------

The chunker converts a StructuredDocument into a
collection of semantic Chunks.

The chunker guarantees that:

1. Every Section produces exactly one Chunk.

2. Every Chunk belongs to exactly one Section.

3. Every Chunk preserves the original business knowledge.

4. Chunk ordering matches the original document order.

The chunker never performs:

- embedding generation
- vector storage
- retrieval
- ranking
"""


#import section
from ml.rag.domain import (
    StructuredDocument,
    Section,
    Chunk
)




def _create_chunk(document: StructuredDocument, section: Section, chunk_index: int) -> Chunk:

            
        # Instantiate the Chunk with structured references
        chunk = Chunk(
            chunk_id=f"chunk_{chunk_index:04d}",
            source_document=document.source_document,
            section=section,
            chunk_index=chunk_index,
            content=section.content
        )


        return chunk
    


#chunk function creates the list of chunk using helper _create_chunk to create a chunk
def chunk(document: StructuredDocument) -> list[Chunk]:

    """
    Convert a StructuredDocument into a list of Chunks.
    """
    chunks = []

    for chunk_index, section in enumerate(document.sections):
        chunks.append(
            _create_chunk(
                document,
                section,
                chunk_index,
            )
        )

    return chunks
