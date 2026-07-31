"""
Biomedical keyword expansion for Elasticsearch full text search.

Expands user queries with synonyms, gene names, disease variations,
and biomedical terminology to improve BM25 retrieval.
"""

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# LLM prompt for biomedical keyword expansion
BIOMEDICAL_KEYWORD_EXPANSION_PROMPT = """You are a biomedical search expert. Your task is to expand a user query
into keywords for Elasticsearch BM25 full-text search.

QUERY: {query}

TASK: Generate {max_keywords} keywords that will help Elasticsearch find relevant biomedical papers using BM25 ranking.

INSTRUCTIONS:
1. Extract key biomedical terms from the query (genes, proteins, diseases, drugs, pathways, cell types)
2. Add synonyms, aliases, and abbreviations commonly used in scientific literature
3. Include both expanded and abbreviated forms (e.g., "BRCA1" AND "breast cancer 1")
4. Focus on terms that appear in biomedical papers and PubMed articles

EXAMPLE:
Query: "What is the role of BRCA1 in DNA repair?"
Output: ["BRCA1", "breast cancer 1", "FANCS", "DNA repair", "double strand break repair",
"DSB repair", "homologous recombination", "HR", "BRCA1 protein", "tumor suppressor"]

OUTPUT FORMAT: Return ONLY a JSON array of keywords. Do not include explanations or markdown formatting.

["keyword1", "keyword2", "keyword3", ...]
"""


async def expand_query_keywords(query: str, llm_provider: Any, config: dict, cache_manager: Any | None = None) -> dict:
    """
    Expand query with biomedical synonyms and terminology.

    Args:
        query: User query string
        llm_provider: LLM provider instance
        config: Global config dict
        cache_manager: Redis cache manager (optional)

    Returns:
        {
            "entities": [{"term": str, "type": str, "synonyms": [str]}],
            "concepts": [{"term": str, "synonyms": [str]}],
            "all_keywords": [str],
            "keyword_string": str,  # Space-separated for ES search
            "cached": bool
        }
    """
    # Get config values
    hybrid_config = config.get("hybrid_search", {})
    max_keywords = hybrid_config.get("max_keywords", 30)
    cache_ttl = hybrid_config.get("keyword_cache_ttl", 86400)
    enable_keyword_expansion = hybrid_config.get("enable_keyword_expansion", True)

    # If expansion disabled, return original query
    if not enable_keyword_expansion:
        return {"entities": [], "concepts": [], "all_keywords": [query], "keyword_string": query, "cached": False}

    # Check cache
    cache_key = None
    if cache_manager:
        cache_key = f"keyword_expansion:{_hash_query(query)}"
        try:
            cached = await cache_manager.get(cache_key)
            if cached:
                logger.info(f"Keyword expansion cache hit for query: {query[:50]}...")
                result = json.loads(cached)
                result["cached"] = True
                return result
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

    # Build prompt
    prompt = BIOMEDICAL_KEYWORD_EXPANSION_PROMPT.format(query=query, max_keywords=max_keywords)

    # Call LLM
    try:
        response = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,  # Deterministic
            max_tokens=1000,
            system_prompt="You are a biomedical search expert. Return ONLY a JSON array of keywords.",
        )

        # Log raw LLM response for debugging
        logger.info(f"Keyword expansion LLM output: {response[:500] if len(response) > 500 else response}")

        # Strip markdown formatting (```json ... ```)
        response_clean = response.strip()
        if response_clean.startswith("```"):
            # Find JSON content between ```json and ```
            lines = response_clean.split("\n")
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not in_code_block and json_lines):
                    json_lines.append(line)
            response_clean = "\n".join(json_lines).strip()

        # Parse JSON array
        keywords_array = json.loads(response_clean)

        # Ensure it's a list
        if not isinstance(keywords_array, list):
            raise ValueError(f"Expected JSON array, got {type(keywords_array)}")

        # Deduplicate and limit
        all_keywords = list(dict.fromkeys(keywords_array))[:max_keywords]

        # Build result
        result = {"all_keywords": all_keywords, "keyword_string": " ".join(all_keywords), "cached": False}

        # Cache result
        if cache_manager and cache_key:
            try:
                await cache_manager.set(cache_key, json.dumps(result), ex=cache_ttl)
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")

        logger.info(f"Keyword expansion (for ES BM25): {len(all_keywords)} keywords")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse keyword expansion JSON: {e}")
        logger.error(f"LLM response: {response[:500] if len(response) > 500 else response}")
        # Re-raise - invalid JSON means LLM failure
        raise ValueError(f"Keyword expansion LLM returned invalid JSON: {e}") from e

    except Exception as e:
        logger.error(f"Keyword expansion failed: {e}")
        import traceback

        logger.error(f"Keyword expansion error: {traceback.format_exc()}")
        # Re-raise to fail the entire pipeline - expansion failure means Elasticsearch won't work
        raise


def _hash_query(query: str) -> str:
    """Generate hash for cache key."""
    return hashlib.md5(query.encode()).hexdigest()
