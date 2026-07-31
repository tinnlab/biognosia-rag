"""
Query expansion for improved chunk retrieval.

Generates focused retrieval queries using LLM to pre-filter chunk candidates
before fetching vectors from Milvus.
"""

import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_EXPANSION_PROMPT = """You are a query reformulation assistant for biomedical information retrieval.

Your task is to reformulate the user's query into multiple focused search queries that capture
different aspects of the information need.

CRITICAL INSTRUCTIONS:
- DO NOT answer the question
- DO NOT explain the biology
- DO NOT invent or add biological knowledge (pathways, processes, functions)
- DO NOT provide analysis
- ONLY rephrase and expand the EXACT terms and concepts already present in the query
- Extract key entities (genes, proteins, diseases, conditions, etc.) and create focused queries
- Each query should use ONLY information explicitly stated in the original query

Original Query: {query}

QUERY EXPANSION STRATEGY:

1. **Query Type Detection** - Identify if this is:
   - Comparison query (e.g., "compare A vs B", "how do A and B differ")
   - Multi-entity query (e.g., "A, B, and C in disease D")
   - Single-entity query (e.g., "What is the role of A")
   - Mechanism query (e.g., "How does A affect B")

2. **Atomic Decomposition** (CRITICAL for multi-entity/comparison queries):
   - For queries with multiple entities (genes, proteins, diseases, conditions):
     * Generate ATOMIC queries focusing on EACH entity INDIVIDUALLY
     * Generate pairwise queries for key interactions
     * Generate comprehensive queries combining entities
   - This ensures retrieval of single-entity papers (most common) AND cross-entity papers (rare)

3. **Quantity Guidelines**:
   - Generate between {min_expansions} and {max_expansions} queries
   - Simple queries: Generate fewer variations (minimum {min_expansions})
   - Complex multi-entity queries: Generate more focused queries (up to {max_expansions})
   - Avoid redundant rephrasing - each query should have distinct retrieval value

EXPANSION EXAMPLES:

Example 1 - Comparison Query:
Input: "How do BRCA1 and BRCA2 differ in DNA repair?"
Output:
["BRCA1 DNA repair mechanisms", "BRCA2 DNA repair mechanisms",
 "BRCA1 versus BRCA2 DNA repair differences", "BRCA1 homologous recombination role",
 "BRCA2 RAD51 interaction in DNA repair"]

Example 2 - Multi-entity Query:
Input: "How do AMPK, mTOR, and ULK1 regulate autophagy?"
Output:
["AMPK regulation of autophagy", "mTOR regulation of autophagy",
 "ULK1 regulation of autophagy", "AMPK activation of ULK1",
 "mTOR inhibition of ULK1", "AMPK mTOR ULK1 autophagy pathway"]

Example 3 - Conditional Query:
Input: "How does TP53 activity differ under DNA damage versus hypoxia?"
Output:
["TP53 activity under DNA damage", "TP53 activity under hypoxia",
 "TP53 activation by DNA damage", "TP53 regulation by hypoxia",
 "TP53 response to genotoxic stress"]

Example 4 - Simple Query:
Input: "What is the role of BRCA1 in DNA repair?"
Output:
["BRCA1 function in DNA repair mechanisms", "BRCA1 involvement in repairing DNA damage",
 "DNA repair role of BRCA1 protein"]

IMPORTANT: You MUST return ONLY a JSON array of strings in this exact format:
["query1", "query2", "query3", ...]

Do NOT return a JSON object like {{"queries": [...]}}.
Do NOT return anything except the JSON array.
Do NOT add explanations before or after the array.

Now generate between {min_expansions} and {max_expansions} focused search queries as a JSON array:"""


async def expand_query_for_retrieval(
    query: str,
    llm_provider,
    num_expansions: int = 2,
    min_expansions: int | None = None,
    max_expansions: int | None = None,
    enable: bool = True,
    prompt_template: str | None = None,
    max_tokens: int = 4000,
    context: str = "",
) -> list[str]:
    """
    Generate expanded queries for chunk retrieval using LLM.

    The original query is ALWAYS included alongside expansions to ensure
    direct match capability while benefiting from semantic variations.

    Supports flexible expansion quantity:
    - If min_expansions and max_expansions are provided, LLM generates between min and max
    - If only num_expansions is provided (legacy), uses num_expansions as both min and max
    - Allows LLM to scale expansions based on query complexity

    Args:
        query: Original user query
        llm_provider: LLM provider instance
        num_expansions: Number of query expansions (legacy, used if min/max not provided)
        min_expansions: Minimum number of expansions to generate (new flexible mode)
        max_expansions: Maximum number of expansions to generate (new flexible mode)
        enable: Whether to enable expansion (fallback to original query)
        prompt_template: Custom prompt template (optional)
        max_tokens: Maximum tokens for LLM response (default: 4000, increased for GPT-5 reasoning)

    Returns:
        List containing [original_query] + expanded queries
        Total queries = variable (min_expansions to max_expansions + 1)
        Example: min_expansions=3, max_expansions=8 returns 4-9 queries total
    """
    if not enable:
        logger.debug("Query expansion disabled, using original query")
        return [query]

    if not llm_provider:
        logger.warning("No LLM provider available, using original query")
        return [query]

    try:
        import time

        start_time = time.time()

        # Determine min/max expansions
        # If min/max not provided, use num_expansions for both (legacy mode)
        if min_expansions is None or max_expansions is None:
            min_exp = num_expansions
            max_exp = num_expansions
            logger.debug(f"Using legacy fixed expansion mode: {num_expansions} queries")
        else:
            min_exp = min_expansions
            max_exp = max_expansions
            logger.debug(f"Using flexible expansion mode: {min_exp}-{max_exp} queries")

        # Use custom template or default
        template = prompt_template or DEFAULT_EXPANSION_PROMPT

        # Truncate query if too long (to avoid max tokens issues)
        # If query contains gene list, sample first N genes
        query_for_expansion = query

        # Be more aggressive with truncation - even 50 genes can be too much
        # Estimate: ~10 chars per gene + commas = ~500 chars for 50 genes
        # Plus template overhead = ~1000 chars total
        # JSON generation needs significant output tokens too
        if len(query) > 2000:  # Lower threshold for long queries
            logger.warning(f"Query too long ({len(query)} chars), truncating for expansion")
            # Try to preserve the question part and sample genes
            if "Here is the gene set:" in query:
                question_part, gene_part = query.split("Here is the gene set:", 1)
                genes = [g.strip() for g in gene_part.split(",") if g.strip()]

                # Dynamically determine sample size based on max_tokens
                # More tokens available = can handle more genes
                # Rule of thumb: max_tokens / 100 = max genes
                max_genes = max(10, min(30, max_tokens // 100))
                sampled_genes = genes[:max_genes]

                query_for_expansion = f"{question_part}Here is the gene set: {', '.join(sampled_genes)}"
                logger.warning(
                    f"Sampled {len(sampled_genes)}/{len(genes)} genes for query expansion "
                    f"(max_tokens={max_tokens}, max_genes={max_genes})"
                )
            else:
                # Just truncate to 2000 chars
                query_for_expansion = query[:2000]
                logger.warning(f"Truncated query to 2000 chars (from {len(query)} chars)")

        # Format prompt with min/max expansions
        prompt = template.format(
            query=query_for_expansion,
            min_expansions=min_exp,
            max_expansions=max_exp,
            num_expansions=num_expansions,  # For backward compatibility if custom templates use it
        )

        # Generate expansions using LLM
        context_str = f" [{context}]" if context else ""
        logger.info(
            f"Generating {min_exp}-{max_exp} query expansions with LLM{context_str} (max_tokens={max_tokens})..."
        )

        # Call LLM (synchronous wrapper for async provider)
        response = await _call_llm(llm_provider, prompt, max_tokens=max_tokens)

        # Log raw LLM response at INFO level for debugging/auditing
        logger.info(f"LLM response (raw content for query expansion):\n{response}")

        # Parse JSON response (now accepts min-max range)
        expanded_queries = _parse_expansion_response(response, query, min_exp, max_exp)

        elapsed = time.time() - start_time
        logger.info(
            f"Query expansion (for Milvus semantic search): generated {len(expanded_queries)} queries in {elapsed:.3f}s"
        )

        # Log expanded queries at INFO level for visibility
        for i, q in enumerate(expanded_queries, 1):
            logger.info(f"  Expansion {i}: {q}")

        return expanded_queries

    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        import traceback

        logger.error(f"Query expansion error: {traceback.format_exc()}")
        # Re-raise to fail the entire pipeline - expansion failure means low quality results
        raise


async def _call_llm(
    llm_provider,
    prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> str:
    """
    Call LLM provider to generate structured JSON array output.

    Args:
        llm_provider: LLM provider instance
        prompt: Formatted prompt
        max_tokens: Maximum tokens for response (default: 4000, increased for GPT-5 reasoning)
        max_retries: Maximum number of retry attempts

    Returns:
        LLM response text (JSON array string)
    """
    import asyncio

    last_error = None

    for attempt in range(max_retries):
        try:
            # All LLM providers have async generate() method
            if hasattr(llm_provider, "generate"):
                # Check if model supports JSON mode (model-based detection)
                # This replaces the old temperature-based heuristic which was incorrect
                should_use_json = llm_provider.supports_json_mode()

                try:
                    response = await llm_provider.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format={"type": "json_object"} if should_use_json else None,
                    )
                    return response
                except Exception as json_error:
                    # Fallback: if response_format causes an error, retry without it
                    error_msg = str(json_error).lower()
                    if "response_format" in error_msg and should_use_json:
                        logger.warning(f"JSON mode not supported despite detection, retrying without: {json_error}")
                        response = await llm_provider.generate(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            response_format=None,
                        )
                        return response
                    else:
                        # Not a response_format error, re-raise
                        raise
            else:
                # Fallback: assume callable (shouldn't happen with standard providers)
                result = llm_provider(prompt, max_tokens=500, temperature=0.3)
                # Check if it's a coroutine and await it
                if hasattr(result, "__await__"):
                    response = await result
                else:
                    response = result
                return response

        except Exception as e:
            last_error = e
            error_msg = str(e)

            # Check if it's a max tokens error
            if "max completion tokens" in error_msg or "json_validate_failed" in error_msg:
                logger.warning(f"Query expansion attempt {attempt + 1}/{max_retries} failed: max tokens reached")
                if attempt < max_retries - 1:
                    # Wait before retry
                    await asyncio.sleep(0.5)
                    continue
            else:
                # Other error - propagate immediately
                raise

    # All retries exhausted
    logger.error(f"Query expansion failed after {max_retries} attempts: {last_error}")
    raise last_error


def _parse_expansion_response(
    response: str, original_query: str, min_expansions: int, max_expansions: int
) -> list[str]:
    """
    Parse LLM response containing query expansions.

    Args:
        response: Raw LLM response
        original_query: Original query (fallback)
        min_expansions: Minimum expected number of expansions
        max_expansions: Maximum expected number of expansions

    Returns:
        List of expanded queries (including original query prepended)
    """
    try:
        # Try to extract JSON from response
        # LLM might return: ```json\n[...]\n``` or just [...]

        # Log raw response for debugging
        logger.debug(f"Raw LLM response (first 500 chars): {response[:500]}")

        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # Parse JSON
        parsed = json.loads(response)

        # Handle both array format and object format
        if isinstance(parsed, list):
            expansions = parsed
        elif isinstance(parsed, dict):
            # JSON mode might wrap in object like {"queries": [...]}
            # Try common keys
            expansions = (
                parsed.get("queries")
                or parsed.get("search_queries")
                or parsed.get("expanded_queries")
                or list(parsed.values())[0]
                if parsed.values()
                else []
            )
        elif isinstance(parsed, str):
            # LLM returned a single string instead of array
            error_msg = f"LLM returned single string instead of JSON array: {parsed[:100]}"
            logger.error(error_msg)
            logger.error(f"Full parsed value: {parsed}")
            raise ValueError(error_msg)
        else:
            error_msg = f"Invalid expansion format: {type(parsed)}"
            logger.error(error_msg)
            logger.error(f"Parsed value: {parsed}")
            raise ValueError(error_msg)

        # Validate format
        if not isinstance(expansions, list):
            error_msg = f"Invalid expansion format (not list): {type(expansions)}"
            logger.error(error_msg)
            logger.error(f"Expansions value: {expansions}")
            raise ValueError(error_msg)

        # Filter to string values only
        expansions = [str(q).strip() for q in expansions if q]

        # Validate non-empty
        if not expansions:
            error_msg = "LLM returned no valid query expansions"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate quantity is within expected range
        num_generated = len(expansions)
        if num_generated < min_expansions:
            logger.warning(
                f"LLM generated {num_generated} expansions, less than minimum {min_expansions}. "
                f"Proceeding with what was generated."
            )
        elif num_generated > max_expansions:
            logger.info(f"LLM generated {num_generated} expansions, limiting to maximum {max_expansions}")
            expansions = expansions[:max_expansions]

        # Always include original query alongside expansions
        # This ensures direct match capability while still benefiting from semantic variations
        if original_query not in expansions:
            # Prepend original query so it's searched first
            expansions = [original_query] + expansions
            logger.debug(f"Added original query to expansions (total: {len(expansions)} queries)")
        else:
            logger.debug("Original query already in expansions")

        return expansions

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from expansion response: {e}")
        logger.error(f"Full LLM response that failed JSON parsing:\n{response}")
        # Re-raise - invalid JSON means LLM failure
        raise ValueError(f"Query expansion LLM returned invalid JSON: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error parsing expansion response: {e}")
        # Re-raise - parsing failure means low quality results
        raise


def get_expansion_cache_key(query: str, num_expansions: int) -> str:
    """
    Generate cache key for query expansions.

    Args:
        query: Original query
        num_expansions: Number of expansions

    Returns:
        Cache key string
    """
    import hashlib

    # Hash query to create stable key
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return f"expansion_{query_hash}_{num_expansions}"


HYDE_PROMPT = """You are a scientific knowledge generation assistant for biomedical information retrieval.

Your task is to generate between {min_hyde} and {max_hyde} DIFFERENT hypothetical answers to the user's question.

CRITICAL INSTRUCTIONS:
- Generate answers in DECLARATIVE language (stating facts, not asking)
- Write as if quoting from research papers
- Focus on biological mechanisms, pathways, and relationships
- Keep each answer to 2-3 sentences
- Generate between {min_hyde} and {max_hyde} distinct hypothetical answers

Original Question: {query}

IMPORTANT: Return a JSON object with this exact format:
{{
  "answers": ["answer1", "answer2", "answer3", ...]
}}

Output ONLY valid JSON:"""


async def generate_hyde(
    query: str,
    llm_provider,
    min_hyde: int = 3,
    max_hyde: int = 5,
    enable: bool = True,
    max_tokens: int = 4000,
    temperature: float = 0.5,
) -> list[str]:
    """Generate HyDE (Hypothetical Document Embeddings) for retrieval.

    Note: max_tokens default is 4000 to accommodate GPT-5's extended reasoning tokens.
    GPT-5 can use 1000-2000+ reasoning tokens for complex queries, so we need sufficient
    budget for both reasoning and the actual JSON response.
    """
    if not enable or not llm_provider:
        return []

    try:
        import time

        start_time = time.time()
        prompt = HYDE_PROMPT.format(query=query, min_hyde=min_hyde, max_hyde=max_hyde)

        logger.info(f"Generating {min_hyde}-{max_hyde} HyDE hypothetical answers...")

        # Call LLM (same as query expansion - automatic request/response logging in provider)
        response = await _call_llm(llm_provider, prompt, max_tokens=max_tokens, temperature=temperature)

        # Log raw LLM response
        logger.info(f"LLM response (raw content for HyDE):\n{response}")

        # Parse JSON response — strip markdown code blocks if present
        response = response.strip()
        if not response:
            logger.warning("LLM returned empty response for HyDE, skipping")
            return []
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        if not response:
            logger.warning("LLM returned empty response for HyDE after stripping, skipping")
            return []
        parsed = json.loads(response)

        # Extract answers from JSON object
        if isinstance(parsed, dict):
            hyde_answers = parsed.get("answers", parsed.get("hypothetical_answers", []))
        else:
            raise ValueError(f"Expected JSON object, got {type(parsed)}")

        hyde_answers = [str(a).strip() for a in hyde_answers if a][:max_hyde]

        if not hyde_answers:
            logger.warning("LLM returned no valid HyDE answers")
            return []

        elapsed = time.time() - start_time
        logger.info(f"HyDE generation: {len(hyde_answers)} answers in {elapsed:.3f}s")

        for i, answer in enumerate(hyde_answers, 1):
            logger.info(f"  HyDE {i}: {answer[:80]}...")

        return hyde_answers

    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        import traceback

        logger.error(f"HyDE traceback:\n{traceback.format_exc()}")
        return []
