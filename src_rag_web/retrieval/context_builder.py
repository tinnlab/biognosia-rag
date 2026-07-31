"""
Context builder for RAG query system.

Formats retrieved data (chunks, entities, relationships) for LLM prompts.

Based on LightRAG operate.py context building functions.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def merge_all_chunks(
    vector_chunks: list[dict[str, Any]],
    entity_chunks: list[dict[str, Any]],
    relationship_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Combine and deduplicate chunks from multiple sources.

    Based on LightRAG operate.py:2984-3084 (_merge_all_chunks)

    Args:
        vector_chunks: Chunks from vector search
        entity_chunks: Chunks from entity-based retrieval
        relationship_chunks: Chunks from relationship-based retrieval

    Returns:
        List of unique chunks with source tags
    """
    logger.debug(
        f"Merging chunks: vector={len(vector_chunks)}, "
        f"entity={len(entity_chunks)}, relationship={len(relationship_chunks)}"
    )

    # Collect all chunks
    all_chunks = []

    # Add vector chunks
    for chunk in vector_chunks:
        chunk_copy = chunk.copy()
        chunk_copy["source"] = "vector"
        all_chunks.append(chunk_copy)

    # Add entity chunks
    for chunk in entity_chunks:
        chunk_copy = chunk.copy()
        chunk_copy["source"] = "entity"
        all_chunks.append(chunk_copy)

    # Add relationship chunks
    for chunk in relationship_chunks:
        chunk_copy = chunk.copy()
        chunk_copy["source"] = "relationship"
        all_chunks.append(chunk_copy)

    # Deduplicate by chunk ID
    seen_ids = set()
    unique_chunks = []

    for chunk in all_chunks:
        chunk_id = chunk.get("id") or chunk.get("chunk_id", "")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)

    logger.info(f"Merged chunks: {len(all_chunks)} total -> {len(unique_chunks)} unique")

    return unique_chunks


def build_query_context(
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Assemble all retrieved data into structured context.

    Based on LightRAG operate.py:3269-3383 (_build_query_context)

    Args:
        chunks: List of text chunks
        entities: List of entities
        relationships: List of relationships

    Returns:
        Dictionary with formatted context sections

    Note:
        Global/hybrid modes not supported (require community detection).
        Use structured entities like Gene Ontology, Pathways instead.
    """
    logger.debug(
        f"Building context: chunks={len(chunks)}, entities={len(entities)}, relationships={len(relationships)}"
    )

    context = {
        "chunks": chunks,
        "entities": entities,
        "relationships": relationships,
    }

    return context


def build_llm_context(
    query: str,
    context: dict[str, Any],
    mode: str = "naive",
    response_type: str = "Multiple Paragraphs",
    min_rerank_score: float = 0.5,
    max_entity_description_length: int = 300,
    entities_only: bool = False,
) -> str:
    """
    Format context data for LLM prompt.

    Based on LightRAG operate.py:3086-3267 (_build_llm_context)

    Args:
        query: User's query
        context: Context dictionary from build_query_context()
        mode: Query mode (naive, local, global, hybrid)
        response_type: Response format
        min_rerank_score: Minimum rerank score threshold for separating chunks

    Returns:
        Formatted prompt string
    """
    from ..llm import PROMPTS
    from ..utils.helpers import convert_to_user_format

    logger.debug(f"Building LLM context: mode={mode}, response_type={response_type}")

    # Select prompt template based on mode
    if mode == "naive":
        prompt_template = PROMPTS["naive_query_prompt"]
    elif mode == "local":
        prompt_template = PROMPTS["local_query_prompt"]
    elif mode == "global":
        logger.error("Global mode not supported (requires community detection). Falling back to local mode.")
        prompt_template = PROMPTS["local_query_prompt"]
    elif mode == "hybrid":
        logger.error("Hybrid mode not supported (requires community detection). Falling back to local mode.")
        prompt_template = PROMPTS["local_query_prompt"]
    elif mode == "mix" and entities_only:
        # Mix mode with entities only (no chunks) - use simplified prompt without citation rules
        prompt_template = PROMPTS["mix_entities_only_prompt"]
    elif mode == "mix":
        # Mix mode combines KG data + vector chunks
        prompt_template = PROMPTS["mix_query_prompt"]
    else:
        logger.error(f"Unknown mode: {mode}, falling back to naive")
        prompt_template = PROMPTS["naive_query_prompt"]

    # Format chunks - split into good vs maybe-related based on rerank score
    chunks = context.get("chunks", [])

    # Get maybe_related_chunks from context (includes early reranking failures from vector search)
    maybe_related_from_context = context.get("maybe_related_chunks", [])

    if mode == "mix" and (chunks or maybe_related_from_context):
        # Split chunks based on rerank score
        good_chunks = []
        maybe_related_chunks_from_main = []

        for chunk in chunks:
            rerank_score = chunk.get("rerank_score", 0.0)
            if rerank_score >= min_rerank_score:
                good_chunks.append(chunk)
            else:
                maybe_related_chunks_from_main.append(chunk)

        # Merge maybe-related chunks from main processing + from context (early reranking failures)
        all_maybe_related = maybe_related_chunks_from_main + maybe_related_from_context

        # Deduplicate by chunk ID
        seen_ids = set()
        unique_maybe_related = []
        for chunk in all_maybe_related:
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_maybe_related.append(chunk)

        # Format each section
        if good_chunks:
            # Sort by chunk ID for document order presentation
            good_chunks.sort(key=lambda x: x.get("id") or x.get("chunk_id", ""))
            good_chunks_formatted = convert_to_user_format(good_chunks)
        else:
            good_chunks_formatted = "No chunks passed rerank threshold."

        if unique_maybe_related:
            # Limit to top 10 by rerank score
            unique_maybe_related.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            unique_maybe_related = unique_maybe_related[:10]
            # Sort by chunk ID for document order presentation
            unique_maybe_related.sort(key=lambda x: x.get("id") or x.get("chunk_id", ""))
            # Continue the handle numbering from the good chunks so every chunk
            # in the prompt has a unique [Cn] handle across both sections.
            maybe_related_formatted = convert_to_user_format(
                unique_maybe_related, start_index=len(good_chunks) + 1
            )
        else:
            maybe_related_formatted = "No additional chunks available."

        logger.info(
            f"Chunk split for prompt: {len(good_chunks)} good chunks (>= {min_rerank_score}), "
            f"{len(unique_maybe_related[:10])} maybe-related chunks (top 10 from {len(all_maybe_related)} total)"
        )
    elif chunks:
        # For non-mix modes, use all chunks as content_data
        # Sort by chunk ID for document order presentation
        chunks.sort(key=lambda x: x.get("id") or x.get("chunk_id", ""))
        content_data = convert_to_user_format(chunks)
        good_chunks_formatted = content_data
        maybe_related_formatted = "No additional chunks available."
    else:
        content_data = "No text chunks available."
        good_chunks_formatted = "No text chunks available."
        maybe_related_formatted = "No additional chunks available."

    # Format entities with hierarchy: system-level first, then molecular-level
    entities = context.get("entities", [])
    if entities:
        # Separate entities by type
        system_entities = []  # KEGG pathways, Reactome pathways
        molecular_entities = []  # Genes, proteins, chemicals, etc.
        other_entities = []  # GO terms, etc.
        skipped_count = 0  # Track entities with empty descriptions

        for entity in entities:
            # Skip entities with empty description
            description = entity.get("description", "").strip()
            if not description:
                skipped_count += 1
                continue

            entity_type = entity.get("entity_type", "Unknown")
            entity_id = entity.get("entity_id", "")

            # Classify as system-level if it's a pathway/system or has MSigDB prefix
            if entity_type in ["KEGG_Pathway", "Reactome_Pathway", "MSigDB_GeneSet"]:
                system_entities.append(entity)
            elif entity_id.startswith("MSigDB:"):
                # Also include entities with MSigDB: prefix in system-level
                system_entities.append(entity)
            elif entity_type in ["Gene", "Protein", "Chemical"]:
                molecular_entities.append(entity)
            else:
                other_entities.append(entity)

        # Log skipped entities
        if skipped_count > 0:
            total_entities = len(system_entities) + len(molecular_entities) + len(other_entities)
            logger.info(
                f"LLM context: included {total_entities} entities, skipped {skipped_count} entities "
                f"with empty descriptions"
            )

        # Use max description length parameter
        max_desc_length = max_entity_description_length

        # Build entity list with clear hierarchy
        entity_lines = []

        # System-level entities first (most important)
        if system_entities:
            entity_lines.append("=== System-Level Entities (Pathways/Systems) ===")
            for i, entity in enumerate(system_entities, 1):
                entity_id = entity.get("entity_id", f"Entity {i}")
                entity_type = entity.get("entity_type", "Unknown")
                description = entity.get("description", "No description")
                # Truncate description if too long
                if max_desc_length > 0 and len(description) > max_desc_length:
                    description = description[:max_desc_length] + "..."
                entity_lines.append(f"[S{i}] {entity_id} ({entity_type}): {description}")
            entity_lines.append("")  # Blank line

        # Molecular-level entities
        if molecular_entities:
            entity_lines.append("=== Molecular-Level Entities (Genes/Proteins/Chemicals) ===")
            for i, entity in enumerate(molecular_entities, 1):
                entity_id = entity.get("entity_id", f"Entity {i}")
                entity_type = entity.get("entity_type", "Unknown")
                description = entity.get("description", "No description")
                # Truncate description if too long
                if max_desc_length > 0 and len(description) > max_desc_length:
                    description = description[:max_desc_length] + "..."
                entity_lines.append(f"[M{i}] {entity_id} ({entity_type}): {description}")
            entity_lines.append("")  # Blank line

        # Other entities (GO terms, etc.)
        if other_entities:
            entity_lines.append("=== Other Entities (Ontology Terms, etc.) ===")
            for i, entity in enumerate(other_entities, 1):
                entity_id = entity.get("entity_id", f"Entity {i}")
                entity_type = entity.get("entity_type", "Unknown")
                description = entity.get("description", "No description")
                # Truncate description if too long
                if max_desc_length > 0 and len(description) > max_desc_length:
                    description = description[:max_desc_length] + "..."
                entity_lines.append(f"[O{i}] {entity_id} ({entity_type}): {description}")

        entity_data = "\n".join(entity_lines)
    else:
        entity_data = "No entities available."

    # Format relationships
    relationships = context.get("relationships", [])
    if relationships:
        relationship_lines = []
        for i, rel in enumerate(relationships, 1):
            src = rel.get("src_id", "?")
            tgt = rel.get("tgt_id", "?")
            description = rel.get("description", "No description")
            relationship_lines.append(f"[{i}] {src} -> {tgt}: {description}")
        relationship_data = "\n".join(relationship_lines)
    else:
        relationship_data = "No relationships available."

    # Fill prompt template
    try:
        if mode == "naive":
            filled_prompt = prompt_template.format(
                query=query,
                content_data=content_data if chunks else "No text chunks available.",
                entity_data=entity_data,
            )
        elif mode == "local":
            filled_prompt = prompt_template.format(
                query=query,
                entity_data=entity_data,
                relationship_data=relationship_data,
            )
        elif mode == "mix" and entities_only:
            # Mix mode entities-only: simplified prompt with just entities
            filled_prompt = prompt_template.format(
                query=query,
                entity_data=entity_data,
            )
        elif mode == "mix":
            # Mix mode: Use split chunks (good vs maybe-related) + entities + relationships
            filled_prompt = prompt_template.format(
                query=query,
                good_chunks=good_chunks_formatted,
                maybe_related_chunks=maybe_related_formatted,
                entity_data=entity_data,
                relationship_data=relationship_data,
            )
        elif mode in ["global", "hybrid"]:
            # Global/hybrid not supported, fall back to local
            filled_prompt = prompt_template.format(
                query=query,
                entity_data=entity_data,
                relationship_data=relationship_data,
            )
        else:
            # Fallback
            filled_prompt = prompt_template.format(
                query=query,
                content_data=content_data if chunks else "No text chunks available.",
                entity_data=entity_data,
            )

        logger.debug(f"LLM context built: {len(filled_prompt)} chars")

        return filled_prompt

    except Exception as e:
        logger.error(f"Failed to build LLM context: {e}")
        import traceback

        logger.debug(traceback.format_exc())

        # Fallback: return fail response prompt
        fail_prompt = PROMPTS.get("fail_response_prompt", "")
        return fail_prompt.format(query=query)


def format_references(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Format chunks as references for the response.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        List of reference dictionaries
    """
    from ..utils.helpers import generate_reference_list_from_chunks

    return generate_reference_list_from_chunks(chunks)
