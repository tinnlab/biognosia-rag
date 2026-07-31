"""
Query decomposition for breaking complex questions into atomic sub-questions.

This module handles structural complexity (breaking multi-part questions into components),
as opposed to query expansion which handles vocabulary diversity (rephrasing the same question).

Examples:
- Comparison: "A vs B?" -> ["original", "What is A?", "What is B?"]
- List: "X in species A, B, C?" -> ["original", "X in A", "X in B", "X in C"]
- Mechanistic: "How do A and B together affect C?" -> ["original", "A->C", "B->C", "A+B interaction"]
- Factual: "What is X?" -> ["original", "X definition"] (minimal decomposition)
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Prompt for query decomposition
DECOMPOSITION_PROMPT = """You are a scientific question analyzer. Your task is to decompose complex scientific questions into atomic sub-questions that can be answered independently.

**Question**: {query}

**Instructions**:
1. Analyze the question type:
   - COMPARISON: Comparing multiple entities, conditions, or mechanisms (e.g., "A vs B", "difference between X and Y")
   - LIST: Requesting information across multiple items (e.g., "how many X in species A, B, C")
   - MECHANISTIC: Multi-component mechanisms or pathways (e.g., "how do A and B together affect C")
   - FACTUAL: Simple, focused question with single answer (e.g., "what is X", "where is Y located")

2. Assess complexity:
   - SIMPLE: Single concept, straightforward answer (minimal decomposition needed)
   - MODERATE: 2-3 main components that can be separated
   - COMPLEX: 4+ components, multiple comparisons, or nested sub-questions

3. Generate atomic sub-questions:
   - ALWAYS include the original question as the first sub-question (preserves full context)
   - For FACTUAL/SIMPLE: Return just the original + 1-2 rephrases
   - For COMPARISON: Create focused queries for each entity/condition being compared
   - For LIST: Create separate queries for each item in the list
   - For MECHANISTIC: Break down individual mechanisms, then combinations
   - Maximum {max_queries} total sub-questions
   - Each sub-question should be answerable independently from a single chunk

**Output Format** (JSON only):
{{
  "question_type": "comparison|list|mechanistic|factual",
  "complexity": "simple|moderate|complex",
  "atomic_questions": [
    "How does salinity affect chlorophyll...",  // Always original first
    "effect of salinity on chlorophyll",
    "salinity impact on plant pigments",
    ...
  ],
  "reasoning": "Brief explanation of decomposition strategy"
}}

**Examples**:

Example 1 - COMPARISON (complex):
Question: "How does salinity affect chlorophyll in tomato vs cucumber, and how does silicon modify these effects?"
Output:
{{
  "question_type": "comparison",
  "complexity": "complex",
  "atomic_questions": [
    "How does salinity affect chlorophyll in tomato vs cucumber, and how does silicon modify these effects?",
    "salinity effect on chlorophyll in tomato",
    "salinity effect on chlorophyll in cucumber",
    "silicon modification of salinity effects on chlorophyll",
    "chlorophyll response to salinity stress",
    "silicon role in plant stress tolerance"
  ],
  "reasoning": "Comparison question with 2 plant species and 2 treatments (salinity, silicon). Decomposed into species-specific queries and treatment-specific queries."
}}

Example 2 - LIST (moderate):
Question: "How many BBX genes in Medicago, peanut, grapevine, pepper?"
Output:
{{
  "question_type": "list",
  "complexity": "moderate",
  "atomic_questions": [
    "How many BBX genes in Medicago, peanut, grapevine, pepper?",
    "BBX genes in Medicago",
    "BBX genes in peanut",
    "BBX genes in grapevine",
    "BBX genes in pepper"
  ],
  "reasoning": "List question across 4 species. Each species needs independent retrieval since answers are likely in different papers."
}}

Example 3 - MECHANISTIC (moderate):
Question: "How do miR-23a and miR-200c influence drug resistance?"
Output:
{{
  "question_type": "mechanistic",
  "complexity": "moderate",
  "atomic_questions": [
    "How do miR-23a and miR-200c influence drug resistance?",
    "miR-23a mechanism in drug resistance",
    "miR-200c mechanism in drug resistance",
    "miRNA regulation of drug resistance"
  ],
  "reasoning": "Mechanistic question with 2 microRNAs. Each miRNA likely has independent mechanism, so separate queries improve targeted retrieval."
}}

Example 4 - FACTUAL (simple):
Question: "What is the role of BRCA1 in DNA repair?"
Output:
{{
  "question_type": "factual",
  "complexity": "simple",
  "atomic_questions": [
    "What is the role of BRCA1 in DNA repair?",
    "BRCA1 function in DNA repair"
  ],
  "reasoning": "Simple factual question. Only minor rephrasing needed for vocabulary diversity."
}}

Now analyze the given question and return ONLY the JSON object (no additional text):"""


async def decompose_query(
    query: str,
    llm_provider,
    max_queries: int = 6,
    enable: bool = True,
    max_tokens: int = 1000,
    temperature: float = 0.3,
) -> list[str]:
    """
    Decompose a complex query into atomic sub-questions.

    Args:
        query: The original query to decompose
        llm_provider: LLM provider instance with generate() method
        max_queries: Maximum number of atomic sub-questions to generate
        enable: Whether decomposition is enabled (if False, returns [query])
        max_tokens: Maximum tokens for LLM response
        temperature: Sampling temperature (lower = more conservative)

    Returns:
        List of atomic sub-questions, always starting with the original query
    """
    if not enable:
        logger.debug("Query decomposition disabled, using original query only")
        return [query]

    if not llm_provider:
        logger.warning("No LLM provider available for query decomposition, using original query only")
        return [query]

    try:
        # Build prompt
        prompt = DECOMPOSITION_PROMPT.format(
            query=query,
            max_queries=max_queries,
        )

        # Generate decomposition
        logger.info(f"Decomposing query: {query[:100]}...")

        # Check if model supports JSON mode (model-based detection)
        should_use_json = llm_provider.supports_json_mode()

        try:
            response = await llm_provider.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"} if should_use_json else None,
            )
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
            else:
                # Not a response_format error, re-raise
                raise

        # Parse JSON response — strip markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response using regex
            logger.warning("Failed to parse JSON directly, trying regex extraction")
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not extract JSON from response")

        # Extract atomic questions
        atomic_questions = data.get("atomic_questions", [])
        question_type = data.get("question_type", "unknown")
        complexity = data.get("complexity", "unknown")
        reasoning = data.get("reasoning", "")

        if not atomic_questions:
            logger.warning("No atomic questions generated, using original query")
            return [query]

        # Ensure original query is first
        if atomic_questions[0] != query:
            logger.warning("Original query not first in atomic questions, prepending it")
            atomic_questions = [query] + atomic_questions

        # Apply max_queries limit
        if len(atomic_questions) > max_queries:
            logger.info(f"Truncating {len(atomic_questions)} atomic questions to max {max_queries}")
            atomic_questions = atomic_questions[:max_queries]

        logger.info(
            f"Query decomposition successful: "
            f"type={question_type}, complexity={complexity}, "
            f"generated {len(atomic_questions)} atomic questions"
        )
        logger.info(f"Reasoning: {reasoning}")
        logger.info("Atomic questions:")
        for i, q in enumerate(atomic_questions, 1):
            logger.info(f"  {i}. {q}")

        return atomic_questions

    except Exception as e:
        logger.error(f"Query decomposition failed: {e}", exc_info=True)
        logger.warning("Falling back to original query only")
        return [query]
