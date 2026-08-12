# Convert retrieved chunk IDs into ordered knowledge units
def chunks_to_knowledge(
    retrieved_chunks: list[str],
    knowledge_map: dict[str, list[str]],
) -> list[str]:

    retrieved_knowledge = []

    for chunk_id in retrieved_chunks:
        concepts = knowledge_map.get(chunk_id, [])
        if isinstance(concepts, list):
                retrieved_knowledge.extend(concepts)  # Use extend() to flatten!
        else:
                retrieved_knowledge.append(concepts)

    return retrieved_knowledge



