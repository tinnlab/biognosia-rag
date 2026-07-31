"""
Helper utilities for RAG query system.

Adapted from plans/lightrag-code/utils/helpers.py
"""

import json
import logging
import re
from hashlib import md5
from typing import Any

logger = logging.getLogger(__name__)


def compute_args_hash(*args) -> str:
    """Compute MD5 hash of arguments

    Args:
        *args: Arguments to hash

    Returns:
        MD5 hash string
    """
    # Concatenate all args into a single string
    combined = "".join(str(arg) for arg in args)
    return md5(combined.encode()).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash ID like LightRAG

    Args:
        content: Content to hash
        prefix: Optional prefix (e.g., "ent-", "rel-")

    Returns:
        Prefixed MD5 hash ID
    """
    return prefix + md5(content.encode()).hexdigest()


def convert_to_user_format(chunks: list[dict], start_index: int = 1) -> str:
    """Convert chunks to user-friendly format

    Each chunk gets a short handle ([C1], [C2], ...) as its header rather than
    its raw `chunk-{40-hex}-{6-digit}` id. Models reproduce a short handle
    reliably but abbreviate a 40-char hash (e.g. `chunk-b080ee8...-000003`),
    which no downstream citation regex can resolve. The handle is stamped back
    onto the chunk dict as `ref_handle` so `generate_reference_list_from_chunks`
    can map it to the real chunk id when resolving footnotes.

    Args:
        chunks: List of chunk dictionaries (mutated: `ref_handle` is set)
        start_index: First handle number, so several sections can share one
            counter (good chunks C1..Cn, then maybe-related Cn+1...)

    Returns:
        Formatted string representation
    """
    if not chunks:
        return "No data available."

    result_parts = []
    for i, chunk in enumerate(chunks, start_index):
        handle = f"C{i}"
        chunk["ref_handle"] = handle
        content = chunk.get("content") or chunk.get("chunk_content", "")
        result_parts.append(f"[{handle}]\n{content}")

    return "\n\n".join(result_parts)


def generate_reference_list_from_chunks(chunks: list[dict]) -> list[dict]:
    """Create reference list from chunks

    Args:
        chunks: List of chunk dictionaries

    Returns:
        List of reference dictionaries
    """
    references = []
    for chunk in chunks:
        ref = {
            "id": chunk.get("id"),
            "content": chunk.get("content") or chunk.get("chunk_content", ""),
            "source_id": chunk.get("source_id"),
            # Short handle the chunk was shown under in the prompt (set by
            # convert_to_user_format). None for chunks the model never saw.
            "handle": chunk.get("ref_handle"),
        }
        # Add optional fields if present
        if "score" in chunk:
            ref["score"] = chunk["score"]
        if "rerank_score" in chunk:
            ref["rerank_score"] = chunk["rerank_score"]

        references.append(ref)

    return references


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers

    Args:
        content: String to split
        markers: List of marker strings

    Returns:
        List of split strings
    """
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value: str) -> bool:
    """Check if string is a valid float

    Args:
        value: String to check

    Returns:
        True if valid float, False otherwise
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def pack_user_ass_to_openai_messages(*args: str):
    """Pack user/assistant messages to OpenAI format

    Args:
        *args: Alternating user/assistant messages

    Returns:
        List of message dictionaries
    """
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(args)]


def load_json(file_name: str):
    """Load JSON from file

    Args:
        file_name: Path to JSON file

    Returns:
        Loaded JSON object or None if file doesn't exist
    """
    import os

    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(json_obj: Any, file_name: str):
    """Write JSON to file

    Args:
        json_obj: Object to serialize
        file_name: Output file path
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from the text

    Remove orphan ...</think> tags from the text also

    Args:
        text: Text to process

    Returns:
        Text with think tags removed
    """
    return re.sub(r"^(<think>.*?</think>|.*</think>)", "", text, flags=re.DOTALL).strip()


def extract_answer_from_tags(text: str) -> tuple[str, str]:
    """Extract final answer from <thinking> and <answer> tags

    This function extracts the concise answer from the LLM response,
    separating internal reasoning from the final user-facing answer.

    Args:
        text: Full LLM response

    Returns:
        Tuple of (final_answer, thinking):
        - final_answer: Content inside <answer> tags (or full text if tags not found)
        - thinking: Content inside <thinking> tags (empty if tags not found)

    Example:
        >>> text = '''<thinking>
        ... Let me analyze the data step by step.
        ... </thinking>
        ... <answer>Yes, the answer is 42.</answer>'''
        >>> answer, thinking = extract_answer_from_tags(text)
        >>> answer
        'Yes, the answer is 42.'
        >>> thinking
        'Let me analyze the data step by step.'
    """
    # Pattern to match <thinking>...</thinking> block (optional)
    thinking_pattern = r"<thinking>(.*?)</thinking>"
    thinking_match = re.search(thinking_pattern, text, re.DOTALL)

    thinking = ""
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        logger.debug(f"Found thinking section: {len(thinking)} chars")

    # Pattern to match <answer>...</answer> block (with closing tag)
    answer_pattern_closed = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern_closed, text, re.DOTALL)

    if answer_match:
        # Found properly closed answer tags
        final_answer = answer_match.group(1).strip()
        logger.debug(f"Extracted answer from tags: answer={len(final_answer)} chars, thinking={len(thinking)} chars")
        return final_answer, thinking

    # Try to find unclosed <answer> tag (LLM may have stopped before closing)
    answer_pattern_unclosed = r"<answer>(.*?)$"
    answer_match_unclosed = re.search(answer_pattern_unclosed, text, re.DOTALL)

    if answer_match_unclosed:
        # Found unclosed answer tag, extract everything after <answer>
        final_answer = answer_match_unclosed.group(1).strip()
        logger.warning(f"Found unclosed <answer> tag, extracted {len(final_answer)} chars")
        logger.debug(
            f"Extracted answer (unclosed tags): answer={len(final_answer)} chars, thinking={len(thinking)} chars"
        )
        return final_answer, thinking

    # No <answer> tags found at all, return full text as answer
    logger.debug("No <answer> tags found in LLM response, using full text")
    return text.strip(), ""
