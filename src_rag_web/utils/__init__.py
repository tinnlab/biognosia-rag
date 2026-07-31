"""Utility functions for RAG query system."""

from .cache import (
    CacheData,
    clear_cache,
    generate_cache_key,
    get_cache_stats,
    handle_cache,
    save_to_cache,
)
from .config_loader import load_config
from .helpers import (
    compute_args_hash,
    compute_mdhash_id,
    convert_to_user_format,
    generate_reference_list_from_chunks,
    is_float_regex,
    load_json,
    pack_user_ass_to_openai_messages,
    remove_think_tags,
    split_string_by_multi_markers,
    write_json,
)
from .text_processing import (
    TiktokenTokenizer,
    Tokenizer,
    cosine_similarity,
    normalize_extracted_info,
    remove_html_tags,
    sanitize_and_normalize_extracted_text,
    sanitize_text_for_encoding,
    truncate_list_by_token_size,
)

__all__ = [
    # Config
    "load_config",
    # Text processing
    "Tokenizer",
    "TiktokenTokenizer",
    "truncate_list_by_token_size",
    "cosine_similarity",
    "sanitize_text_for_encoding",
    "normalize_extracted_info",
    "remove_html_tags",
    "sanitize_and_normalize_extracted_text",
    # Helpers
    "compute_args_hash",
    "compute_mdhash_id",
    "convert_to_user_format",
    "generate_reference_list_from_chunks",
    "split_string_by_multi_markers",
    "is_float_regex",
    "pack_user_ass_to_openai_messages",
    "load_json",
    "write_json",
    "remove_think_tags",
    # Cache
    "CacheData",
    "generate_cache_key",
    "handle_cache",
    "save_to_cache",
    "clear_cache",
    "get_cache_stats",
]
