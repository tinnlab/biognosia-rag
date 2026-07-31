"""
Entity extraction module using n-gram matching.

This module provides n-gram based entity extraction for RAG queries,
replacing LightRAG's keyword extraction with more precise entity matching.
"""

from .entity_statistics import EntityStatisticsManager
from .ngram_matcher import NGramEntityMatcher, NGramGenerator

__all__ = [
    "NGramGenerator",
    "NGramEntityMatcher",
    "EntityStatisticsManager",
]
