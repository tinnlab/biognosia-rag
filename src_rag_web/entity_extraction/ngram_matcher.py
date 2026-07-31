"""
N-gram entity matching for query entity extraction.

Uses sophisticated n-gram generation with POS filtering, stop word removal,
and quality filters to extract only entity-like phrases.
"""

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


@dataclass
class NGram:
    """N-gram with position information."""

    text: str
    start: int  # Start character position
    end: int  # End character position
    k: int  # N-gram size (number of tokens)
    pos_tags: list = None  # POS tags for tokens (optional, for filtering)


class NGramGenerator:
    """
    Generate n-grams from text with sentence boundaries and sophisticated filtering.

    Filters out non-entity phrases using:
    - POS-based filtering (determiners, prepositions, conjunctions, verbs)
    - Stop word removal
    - Pronoun filtering
    - "To be" verb filtering
    - Past tense verb filtering
    - Comparison/temporal word filtering
    - Boundary word validation
    """

    def __init__(
        self,
        k_values: list[int] = None,
        remove_stopwords: bool = True,
        filter_pos: bool = True,
        language: str = "english",
    ):
        """
        Initialize n-gram generator.

        Args:
            k_values: List of n-gram sizes (number of tokens)
            remove_stopwords: Whether to filter out n-grams containing only stop words
            filter_pos: Whether to filter n-grams based on POS tags
            language: Language for stop words (default: 'english')
        """
        self.k_values = sorted(k_values or [1, 2, 3, 4, 5])
        self.remove_stopwords = remove_stopwords
        self.filter_pos = filter_pos

        # Load stop words
        if remove_stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set()

        # POS tags to filter from n-gram boundaries
        # These are non-entity words that shouldn't start or end n-grams
        self.invalid_start_tags = {
            "CC",  # Coordinating conjunction (and, or, but)
            "IN",  # Preposition/subordinating conjunction (of, in, on, at)
            "DT",  # Determiner (the, a, an, this, that)
            "TO",  # "to"
            "WDT",  # Wh-determiner (which, that)
            "WP",  # Wh-pronoun (who, what)
            "WRB",  # Wh-adverb (when, where, how)
            "RB",  # Adverb (very, also, however)
            "JJ",  # Adjective (such, many, various)
            "JJR",  # Adjective, comparative (better, larger)
            "JJS",  # Adjective, superlative (best, largest)
            "VB",  # Verb, base form
            "VBD",  # Verb, past tense
            "VBG",  # Verb, gerund/present participle
            "VBN",  # Verb, past participle
            "VBP",  # Verb, non-3rd person singular present
            "VBZ",  # Verb, 3rd person singular present
            "MD",  # Modal verb (can, could, will, would, should, may, might, must)
            "CD",  # Cardinal number (one, two, 1, 2)
            "PRP",  # Personal pronoun (I, you, it, he)
            "PRP$",  # Possessive pronoun (my, your, its)
        }

        self.invalid_end_tags = {
            "CC",  # Coordinating conjunction (and, or, but)
            "IN",  # Preposition/subordinating conjunction (of, in, on, at)
            "DT",  # Determiner (the, a, an)
            "TO",  # "to"
            "VB",  # Verb, base form
            "VBD",  # Verb, past tense
            "VBG",  # Verb, gerund/present participle
            "VBN",  # Verb, past participle
            "VBP",  # Verb, non-3rd person singular present
            "VBZ",  # Verb, 3rd person singular present
            "MD",  # Modal verb (can, could, will, would, should, may, might, must)
            "RB",  # Adverb (very, also, however)
            "CD",  # Cardinal number (one, two, 1, 2)
            "PRP",  # Personal pronoun (I, you, it, he)
            "PRP$",  # Possessive pronoun (my, your, its)
        }

        # Invalid words at boundaries (case-insensitive)
        self.invalid_start_words = {
            "that",
            "it",
            "which",
            "where",
            "whereare",
            "there",
            "they",
            "these",
            "those",
            "this",
            "what",
            "when",
            "who",
            "whereas",
            "would",
            "will",
            "could",
            "can",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "ought",
        }

        self.invalid_end_words = {
            "that",
            "it",
            "which",
            "where",
            "whereare",
            "there",
            "they",
            "these",
            "those",
            "this",
            "what",
            "when",
            "who",
            "whereas",
            "would",
            "will",
            "could",
            "can",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "ought",
            "such",
            "as",
        }

        # Forms of "to be" verb
        self.to_be_forms = {"is", "are", "was", "were", "am", "be", "been", "being"}

        # Pronouns that should not appear anywhere in n-grams
        self.pronouns = {
            # Personal pronouns
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            # Possessive pronouns
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "hers",
            "ours",
            "theirs",
            # Reflexive pronouns
            "myself",
            "yourself",
            "himself",
            "herself",
            "itself",
            "ourselves",
            "yourselves",
            "themselves",
            # Demonstrative pronouns
            "this",
            "that",
            "these",
            "those",
            # Interrogative/relative pronouns
            "which",
            "who",
            "whom",
            "whose",
            "what",
            "where",
            "when",
            "why",
            "how",
        }

        # Comparison/temporal words that should not appear anywhere
        self.comparison_temporal_words = {"then", "than", "more", "less", "most", "least"}

        # POS tags for comparative/superlative adjectives to filter
        self.comparative_tags = {"JJR", "JJS", "RBR", "RBS"}

        # POS tags for past tense verbs to filter
        self.past_tense_tags = {"VBD", "VBN"}  # VBD: past tense, VBN: past participle

    def generate(self, text: str) -> list[NGram]:
        """
        Generate n-grams from text with sentence-aware tokenization.

        This method:
        1. Protects single-letter abbreviations (e.g., "S. aureus")
        2. Splits text into sentences
        3. Tokenizes each sentence using NLTK
        4. Generates n-grams within sentence boundaries
        5. Filters out n-grams containing only stop words
        6. Applies POS-based filtering
        7. Maintains accurate character positions

        Args:
            text: Input text

        Returns:
            List of NGram objects with text and position info
        """
        ngrams = []

        # Protect single-letter abbreviations by temporarily replacing them
        # Pattern: Single capital letter + period + space + lowercase letter
        # Examples: "S. aureus", "P. aeruginosa", "C. longa"
        abbrev_pattern = r"\b([A-Z])\.\s+([a-z])"
        abbrev_placeholder = r"\1__ABBREV__\2"
        text_protected = re.sub(abbrev_pattern, abbrev_placeholder, text)

        # Split into sentences
        sentences = sent_tokenize(text_protected)

        # Track current position in original text
        current_pos = 0

        for sentence_protected in sentences:
            # Restore abbreviations in sentence
            sentence = re.sub(r"([A-Z])__ABBREV__([a-z])", r"\1. \2", sentence_protected)

            # Find sentence position in original text
            sentence_start = text.find(sentence, current_pos)
            if sentence_start == -1:
                continue
            current_pos = sentence_start

            # Tokenize sentence and track positions
            tokens_with_pos = []

            # Use word_tokenize which respects punctuation
            words = word_tokenize(sentence)

            # Find each word's position in the original text
            search_start = sentence_start
            for word in words:
                # Skip pure punctuation tokens
                if re.match(r"^[\W_]+$", word):
                    continue

                # Find word position in original text
                word_pos = text.find(word, search_start)
                if word_pos != -1:
                    tokens_with_pos.append((word, word_pos, word_pos + len(word)))
                    search_start = word_pos + len(word)

            # Generate n-grams within this sentence
            for k in self.k_values:
                if k > len(tokens_with_pos):
                    continue

                # Use NLTK's ngrams utility
                for i in range(len(tokens_with_pos) - k + 1):
                    ngram_tokens = tokens_with_pos[i : i + k]

                    # Extract tokens and positions
                    tokens = [t[0] for t in ngram_tokens]
                    start = ngram_tokens[0][1]
                    end = ngram_tokens[-1][2]
                    ngram_text = text[start:end].strip()

                    # Skip if text became empty after stripping
                    if not ngram_text:
                        continue

                    # Skip n-grams containing newlines (spanning multiple lines)
                    if "\n" in ngram_text or "\r" in ngram_text:
                        continue

                    # N-grams must start and end with alphanumeric characters only
                    # No dashes, hashes, or other special characters at boundaries
                    if not ngram_text[0].isalnum() or not ngram_text[-1].isalnum():
                        continue

                    # Filter out stop-word-only n-grams
                    if self.remove_stopwords:
                        # Skip if all tokens are stop words
                        if all(t.lower() in self.stop_words for t in tokens):
                            continue

                    # Skip very short n-grams (require at least 3 letters)
                    if len(ngram_text) < 3:
                        continue

                    # Skip pure numbers
                    if re.match(r"^[\d\W]+$", ngram_text):
                        continue

                    # Skip n-grams that are just single characters
                    if all(len(t) == 1 for t in tokens):
                        continue

                    # Only allow specific characters: alphabets, numbers, dash, underscore, slash, dot, and spaces
                    # Disallow: =, >, <, ?, %, commas, colons, semicolons, parentheses, brackets, etc.
                    if not re.match(r"^[a-zA-Z0-9\-_/. ]+$", ngram_text):
                        continue

                    # Skip n-grams ending with period
                    if ngram_text.endswith("."):
                        continue

                    # POS-based filtering: check start and end tokens
                    # Store POS tags for later use in post-filtering
                    pos_tags = None
                    if self.filter_pos:
                        # Get POS tags for tokens
                        pos_tags = pos_tag(tokens)

                        # Check first token POS tag (skip JJ check for now, will post-filter)
                        invalid_start_without_jj = self.invalid_start_tags - {"JJ"}
                        if pos_tags[0][1] in invalid_start_without_jj:
                            continue

                        # Check last token POS tag
                        if pos_tags[-1][1] in self.invalid_end_tags:
                            continue

                        # Check first token word (case-insensitive)
                        if tokens[0].lower() in self.invalid_start_words:
                            continue

                        # Check last token word (case-insensitive)
                        if tokens[-1].lower() in self.invalid_end_words:
                            continue

                        # Filter out n-grams containing "to be" verbs anywhere
                        # Examples: "bacteria are resistant", "cells were tested", "biofilm is formed"
                        if any(token.lower() in self.to_be_forms for token in tokens):
                            continue

                        # Filter out n-grams containing pronouns anywhere
                        # Examples: "its activity", "they tested", "we observed", "her results"
                        if any(token.lower() in self.pronouns for token in tokens):
                            continue

                        # Filter out n-grams containing comparison/temporal words
                        # Examples: "then observed", "bigger than", "more effective", "most common"
                        if any(token.lower() in self.comparison_temporal_words for token in tokens):
                            continue

                        # Filter out n-grams containing comparative/superlative adjectives or adverbs
                        # Examples: "bigger samples", "highest concentration", "better results", "faster growth"
                        if any(pos_tag[1] in self.comparative_tags for pos_tag in pos_tags):
                            continue

                        # Filter out n-grams containing past tense verbs
                        # Examples: "samples tested", "cells treated", "bacteria showed", "extract demonstrated"
                        if any(pos_tag[1] in self.past_tense_tags for pos_tag in pos_tags):
                            continue

                    # Store n-gram with POS tags for post-filtering
                    ngram_obj = NGram(text=ngram_text, start=start, end=end, k=k)
                    # Attach POS tags to ngram object for post-filtering
                    if pos_tags:
                        ngram_obj.pos_tags = pos_tags
                    ngrams.append(ngram_obj)

            # Update current position for next sentence
            current_pos = sentence_start + len(sentence)

        # Post-filter: Remove 1-gram adjectives (JJ) that appear in longer n-grams
        # This allows isolated entity names like "Fallopian" while filtering common adjectives
        if self.filter_pos and ngrams:
            # Build set of all words that appear in n-grams >= 2
            words_in_longer_ngrams = set()
            for ng in ngrams:
                if ng.k >= 2:
                    # Extract individual words from this n-gram
                    words_in_longer_ngrams.update(ng.text.split())

            # Filter 1-grams: remove JJ-tagged 1-grams that appear in longer n-grams
            filtered_ngrams = []
            for ng in ngrams:
                # Keep all n-grams >= 2
                if ng.k >= 2:
                    filtered_ngrams.append(ng)
                    continue

                # For 1-grams, check if tagged as JJ and appears in longer n-grams
                if ng.pos_tags and len(ng.pos_tags) > 0:
                    first_tag = ng.pos_tags[0][1]
                    # If 1-gram is JJ and appears in a longer n-gram, filter it out
                    if first_tag == "JJ" and ng.text in words_in_longer_ngrams:
                        continue

                # Keep this 1-gram (either not JJ, or JJ but isolated)
                filtered_ngrams.append(ng)

            ngrams = filtered_ngrams

        return ngrams

    def generate_ngrams(self, text: str) -> set[str]:
        """
        Generate unique n-grams from text (returns set of strings).

        Convenience method that wraps generate() and returns a set of strings
        instead of NGram objects with position info.

        Args:
            text: Input text

        Returns:
            Set of unique n-gram strings
        """
        ngram_objects = self.generate(text)

        # Log breakdown by k value
        from collections import Counter

        k_counts = Counter(ng.k for ng in ngram_objects)
        breakdown = ", ".join(f"{k}-grams: {count}" for k, count in sorted(k_counts.items()))
        if breakdown:
            logger.info(f"N-gram breakdown by size: {breakdown}")

        return {ng.text for ng in ngram_objects}


class NGramEntityMatcher:
    """Match n-grams to entities using vector similarity."""

    def __init__(
        self,
        embedding_manager,
        milvus_storage,
        k_values: list[int] = None,
        entity_stats_manager=None,
        max_pct_paper: float | None = None,
        max_pct_chunk: float | None = None,
        max_num_text_variations: int | None = None,
        min_similarity: float = 0.85,
        entity_collections: list[str] | None = None,
        enable_cache: bool = True,
        max_cache_size: int = 10000,
        enable_stage2: bool = False,
        stage2_config: dict = None,
    ):
        """
        Initialize n-gram entity matcher.

        Args:
            embedding_manager: Embedding manager from src_rag
            milvus_storage: Milvus storage from src_rag (handles entity collection discovery)
            k_values: N-gram sizes to generate (default: [1, 2, 3, 4, 5])
            entity_stats_manager: Entity statistics manager for filtering
            max_pct_paper: Maximum pct_paper threshold
            max_pct_chunk: Maximum pct_chunk threshold
            max_num_text_variations: Maximum num_text_variations threshold
            min_similarity: Minimum cosine similarity threshold for entity matching (default: 0.85)
            entity_collections: List of entity collections to search (Stage 1 only, e.g., ["Genes", "Disease"])
            enable_cache: Enable n-gram caching
            max_cache_size: Maximum cache size (number of n-grams)
            enable_stage2: Enable second-stage semantic community discovery
            stage2_config: Configuration dict for Stage 2 discovery
        """
        self.embedding_manager = embedding_manager
        self.milvus_storage = milvus_storage
        self.ngram_generator = NGramGenerator(k_values)
        self.entity_stats_manager = entity_stats_manager
        self.max_pct_paper = max_pct_paper
        self.max_pct_chunk = max_pct_chunk
        self.max_num_text_variations = max_num_text_variations
        self.min_similarity = min_similarity
        self.entity_collections = entity_collections

        # Caching
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.positive_cache = OrderedDict()  # ngram_text -> list[entity_dict]
        self.negative_cache = set()  # Set[ngram_text] with no matches

        # Stage 2: Semantic community discovery
        self.enable_stage2 = enable_stage2
        self.stage2_config = stage2_config or {}
        self.community_discoverer = None

        if enable_stage2:
            from .semantic_community import SemanticCommunityDiscovery

            self.community_discoverer = SemanticCommunityDiscovery(
                embedding_manager=embedding_manager, milvus_storage=milvus_storage, **stage2_config
            )

    async def match_query_entities(self, query: str, detailed_logger=None) -> tuple[list[dict], dict]:
        """
        Extract entities from query using n-gram matching (Stage 1 + optional Stage 2).

        Args:
            query: User query text
            detailed_logger: Optional DetailedLogger for structured logging

        Returns:
            Tuple of (all_entities, metadata)
            - all_entities: Combined Stage 1 + Stage 2 entities
            - metadata: Dict with stage1_count, stage2_count, etc.
        """
        if not query:
            logger.warning("Empty query provided")
            return [], {"stage1_count": 0, "stage2_count": 0, "total_count": 0}

        # Stage 1: N-gram matching with snippet tracking
        stage1_entities, entity_text_snippets, stage1_matched, stage1_filtered = await self._match_stage1(
            query, detailed_logger
        )

        # Stage 2: Semantic community discovery (optional)
        stage2_entities = []
        if self.enable_stage2 and self.community_discoverer:
            min_entities = self.stage2_config.get("min_entities", 2)
            if len(stage1_entities) >= min_entities:
                logger.info(f"Stage 2 enabled: {len(stage1_entities)} Stage 1 entities found")
                stage2_entities = await self.community_discoverer.discover_entities(
                    stage1_entities, entity_text_snippets, detailed_logger
                )
            else:
                logger.info(
                    f"Stage 2 skipped: only {len(stage1_entities)} Stage 1 entities (minimum required: {min_entities})"
                )
        else:
            # Log why Stage 2 is disabled
            if logger.isEnabledFor(logging.DEBUG):
                if not self.enable_stage2:
                    logger.debug("Stage 2: Disabled (enable_stage2=False in configuration)")
                elif not self.community_discoverer:
                    logger.debug("Stage 2: No community discoverer initialized")

        # Combine and deduplicate
        all_entities = stage1_entities + stage2_entities

        metadata = {
            "stage1_count": len(stage1_entities),
            "stage2_count": len(stage2_entities),
            "total_count": len(all_entities),
        }

        logger.info(
            f"Entity extraction: {metadata['stage1_count']} direct, "
            f"{metadata['stage2_count']} discovered, "
            f"{metadata['total_count']} total"
        )

        # Log entity names at INFO level
        if stage1_entities:
            stage1_names = [e.get("entity_name", "Unknown") for e in stage1_entities[:10]]
            logger.info(f"Stage 1 entities (showing first 10): {', '.join(stage1_names)}")
            if len(stage1_entities) > 10:
                logger.info(f"  ... and {len(stage1_entities) - 10} more Stage 1 entities")

        if stage2_entities:
            stage2_names = [e.get("entity_name", "Unknown") for e in stage2_entities[:10]]
            logger.info(f"Stage 2 entities (showing first 10): {', '.join(stage2_names)}")
            if len(stage2_entities) > 10:
                logger.info(f"  ... and {len(stage2_entities) - 10} more Stage 2 entities")

        # Detailed logging: log entities final summary
        if detailed_logger:
            # Build final entities list with structured info
            final_entities_list = []
            for entity in all_entities:
                entity_name = entity.get("entity_name", "unknown")
                entity_type = entity.get("entity_type")

                # If entity_type not set, extract from prefix (GENE:xxx → Gene)
                if not entity_type or entity_type == "unknown":
                    if ":" in entity_name:
                        prefix = entity_name.split(":")[0]
                        # Map prefix to type
                        type_map = {
                            "GENE": "Gene",
                            "GO": "GeneOntology",
                            "KEGG": "KEGGPathway",
                            "DISEASE": "Disease",
                            "CHEM": "Chemical",
                            "MESH": "Disease",
                            "CELL": "CellOntology",
                        }
                        entity_type = type_map.get(prefix, prefix)
                    else:
                        entity_type = "unknown"

                final_entities_list.append({"id": entity_name, "type": entity_type, "name": entity_name})

            detailed_logger.log_entities_final(
                {
                    "stage1_matched": stage1_matched,
                    "stage1_filtered": stage1_filtered,
                    "stage1_kept": metadata["stage1_count"],
                    "stage2_discovered": metadata["stage2_count"],
                    "final_entities": final_entities_list,
                }
            )

        return all_entities, metadata

    async def _match_stage1(
        self, query: str, detailed_logger=None
    ) -> tuple[list[dict], dict[str, list[str]], int, int]:
        """
        Stage 1 matching with text snippet tracking.

        Args:
            query: User query text
            detailed_logger: Optional DetailedLogger for structured logging

        Returns:
            Tuple of (entities, entity_text_snippets, stage1_matched, stage1_filtered)
            - entities: Matched entity dicts
            - entity_text_snippets: Mapping of entity_name -> [ngram1, ngram2, ...]
            - stage1_matched: Total number of matched entities before filtering
            - stage1_filtered: Number of entities filtered out
        """
        import time

        stage1_start = time.time()

        # Step 1: Generate unique n-grams
        ngram_start = time.time()
        ngrams = self.ngram_generator.generate_ngrams(query)
        ngram_elapsed = time.time() - ngram_start
        logger.info(f"Stage 1 timing: N-gram generation: {len(ngrams)} n-grams in {ngram_elapsed:.3f}s")

        # Print all n-grams generated
        logger.info("All generated n-grams:")
        for i, ngram in enumerate(sorted(ngrams), 1):
            logger.info(f"  {i}. '{ngram}'")

        if not ngrams:
            return [], {}, 0, 0

        # Step 2: Collect entities from cache and new matches
        # Track which n-grams matched which entities
        from collections import defaultdict

        ngram_to_entities = defaultdict(list)
        all_entities = []
        seen_entity_names = set()

        # Check cache first
        cache_start = time.time()
        uncached_ngrams = set()
        for ngram in ngrams:
            if self.enable_cache:
                if ngram in self.positive_cache:
                    # Cache hit - add entities
                    for entity in self.positive_cache[ngram]:
                        entity_name = entity.get("entity_name")
                        if entity_name:
                            ngram_to_entities[ngram].append(entity_name)
                            if entity_name not in seen_entity_names:
                                all_entities.append(entity)
                                seen_entity_names.add(entity_name)
                    continue
                elif ngram in self.negative_cache:
                    # Known non-match
                    continue

            # Not in cache - needs matching
            uncached_ngrams.add(ngram)

        cache_elapsed = time.time() - cache_start
        cache_hits = len(ngrams) - len(uncached_ngrams)
        logger.info(
            f"Stage 1 timing: Cache lookup: {cache_hits} hits, {len(uncached_ngrams)} misses in {cache_elapsed:.3f}s"
        )

        # Step 3: Match uncached n-grams and track snippets
        if uncached_ngrams:
            match_start = time.time()
            new_matches, new_ngram_to_entities = await self._match_ngrams_with_tracking(
                uncached_ngrams, detailed_logger
            )
            match_elapsed = time.time() - match_start

            logger.info(
                f"Stage 1 timing: N-gram matching: {len(uncached_ngrams)} n-grams -> "
                f"{len(new_matches)} entities in {match_elapsed:.3f}s"
            )

            # Update mappings
            for ngram, entity_names in new_ngram_to_entities.items():
                ngram_to_entities[ngram].extend(entity_names)

            # Add new entities
            for entity in new_matches:
                entity_name = entity.get("entity_name")
                if entity_name and entity_name not in seen_entity_names:
                    all_entities.append(entity)
                    seen_entity_names.add(entity_name)

        # Step 4: Filter by entity statistics
        filter_start = time.time()
        stage1_matched_count = len(all_entities)  # Track before filtering
        filtered_entities = self._filter_by_statistics(all_entities)
        stage1_kept_count = len(filtered_entities)  # Track after filtering
        stage1_filtered_count = stage1_matched_count - stage1_kept_count
        filter_elapsed = time.time() - filter_start
        logger.info(
            f"Stage 1 timing: Entity filtering: {stage1_matched_count} -> {stage1_kept_count} entities "
            f"in {filter_elapsed:.3f}s"
        )

        # Log which entities were filtered out with their statistics
        if len(filtered_entities) < len(all_entities):
            filtered_out_entities = [
                e
                for e in all_entities
                if e.get("entity_name") not in [fe.get("entity_name") for fe in filtered_entities]
            ]
            logger.info(f"  Filtered out {len(filtered_out_entities)} entities (too common):")

            # Show first 10 with statistics
            for entity in filtered_out_entities[:10]:
                entity_name = entity.get("entity_name", "Unknown")
                stats_str = ""
                if self.entity_stats_manager:
                    stats = self.entity_stats_manager.get_statistics(entity_name)
                    if stats:
                        pct_paper = stats.get("pct_paper", 0.0)
                        pct_chunk = stats.get("pct_chunk", 0.0)
                        num_texts = stats.get("num_text_variations")

                        stats_str = f" (pct_paper={pct_paper:.1f}%, pct_chunk={pct_chunk:.1f}%"
                        if num_texts is not None:
                            stats_str += f", num_texts={num_texts}"
                        stats_str += ")"

                logger.info(f"    {entity_name}{stats_str}")

            if len(filtered_out_entities) > 10:
                logger.info(f"    ... and {len(filtered_out_entities) - 10} more")

        # Step 5: Build entity -> snippets mapping (invert ngram_to_entities)
        from collections import defaultdict

        entity_text_snippets = defaultdict(list)
        for ngram, entity_names in ngram_to_entities.items():
            for entity_name in entity_names:
                # Only include snippets for entities that passed filtering
                if any(e.get("entity_name") == entity_name for e in filtered_entities):
                    entity_text_snippets[entity_name].append(ngram)

        stage1_elapsed = time.time() - stage1_start
        logger.info(
            f"Stage 1: Matched {stage1_matched_count} entities, {stage1_kept_count} after filtering "
            f"(total: {stage1_elapsed:.3f}s)"
        )

        return filtered_entities, dict(entity_text_snippets), stage1_matched_count, stage1_filtered_count

    async def _match_ngrams(self, ngrams: set[str]) -> list[dict]:
        """
        Match n-grams to entities across all entity collections.

        Args:
            ngrams: Set of n-gram strings to match

        Returns:
            List of matched entity dicts with full metadata
        """
        if not ngrams:
            return []

        # Convert to list for batch processing
        ngram_list = list(ngrams)

        # Step 1: Compute embeddings for all n-grams (batch)
        try:
            embeddings = await self._compute_embeddings_batch(ngram_list)
        except Exception as e:
            logger.error(f"CRITICAL: Failed to compute embeddings: {e}")
            logger.error("Stopping query pipeline due to embedding computation failure")
            raise

        # Step 2: Search ALL entity collections (MilvusStorage handles discovery & min_similarity)
        try:
            all_entities = await self._search_entities(ngram_list, embeddings)
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            all_entities = []

        # Step 3: Update cache
        if self.enable_cache:
            self._update_cache(ngram_list, all_entities)

        return all_entities

    async def _match_ngrams_with_tracking(
        self, ngrams: set[str], detailed_logger=None
    ) -> tuple[list[dict], dict[str, list[str]]]:
        """
        Match n-grams to entities with tracking of which n-grams matched which entities.

        Args:
            ngrams: Set of n-gram strings to match
            detailed_logger: Optional DetailedLogger for structured logging

        Returns:
            Tuple of (entities, ngram_to_entities)
            - entities: List of matched entity dicts
            - ngram_to_entities: Dict mapping ngram_text -> [entity_name1, entity_name2, ...]
        """
        if not ngrams:
            return [], {}

        import time
        from collections import defaultdict

        # Convert to list for batch processing
        ngram_list = list(ngrams)

        # Step 1: Compute embeddings for all n-grams (batch)
        embed_start = time.time()
        try:
            embeddings = await self._compute_embeddings_batch(ngram_list)
            embed_elapsed = time.time() - embed_start
            logger.info(f"  Embedding computation: {len(ngram_list)} n-grams in {embed_elapsed:.3f}s")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to compute embeddings: {e}")
            logger.error("Stopping query pipeline due to embedding computation failure")
            raise

        # Step 2: Search entity collections (per n-gram results)
        search_start = time.time()
        try:
            results = await self.milvus_storage.search_entities(
                query_embeddings=embeddings,
                top_k=1,  # Top 1 match per n-gram
                min_similarity=self.min_similarity,
                include_collections=self.entity_collections,
            )
            search_elapsed = time.time() - search_start
            logger.info(f"  Entity search (Milvus): {len(ngram_list)} queries in {search_elapsed:.3f}s")
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            results = []

        # Step 3: Build mappings and collect entities
        process_start = time.time()
        ngram_to_entities = defaultdict(list)
        all_entities = []
        seen_entity_names = set()

        # Track n-gram -> (entity, score) for detailed logging
        ngram_match_details = []

        for ngram, result_list in zip(ngram_list, results):
            for result in result_list:
                entity_name = result.get("entity_name")
                score = result.get("score", 0.0)
                if entity_name:
                    # Track mapping
                    ngram_to_entities[ngram].append(entity_name)

                    # Store match details for logging
                    ngram_match_details.append({"ngram": ngram, "entity": entity_name, "score": score})

                    # Collect unique entities
                    if entity_name not in seen_entity_names:
                        all_entities.append(result)
                        seen_entity_names.add(entity_name)

        process_elapsed = time.time() - process_start
        logger.info(f"  Result processing: {len(all_entities)} unique entities in {process_elapsed:.3f}s")

        # Log detailed n-gram matching results with entity statistics and filter status
        if ngram_match_details:
            logger.info(f"  N-gram match details ({len(ngram_match_details)} matches):")
            for match in ngram_match_details:
                entity_name = match["entity"]
                score = match["score"]

                # Get entity statistics if available
                stats_str = ""
                filter_status = "KEPT"
                filter_reason = None
                pct_paper = 0.0
                pct_chunk = 0.0

                if self.entity_stats_manager:
                    stats = self.entity_stats_manager.get_statistics(entity_name)
                    if stats:
                        pct_paper = stats.get("pct_paper", 0.0)
                        pct_chunk = stats.get("pct_chunk", 0.0)
                        num_texts = stats.get("num_text_variations")

                        # Build stats string
                        stats_str = f", pct_paper={pct_paper:.1f}%, pct_chunk={pct_chunk:.1f}%"
                        if num_texts is not None:
                            stats_str += f", num_texts={num_texts}"

                        # Check filter status
                        filter_reason = self.entity_stats_manager.get_filter_reason(
                            entity_name, self.max_pct_paper, self.max_pct_chunk, self.max_num_text_variations
                        )
                        if filter_reason:
                            filter_status = "FILTERED"
                    else:
                        stats_str = ", stats=N/A (high quality)"

                # Detailed logging: log entity stage1 match
                if detailed_logger:
                    detailed_logger.log_entity_stage1_match(
                        {
                            "ngram": match["ngram"],
                            "entity_id": entity_name,
                            "score": float(score),
                            "pct_paper": float(pct_paper),
                            "pct_chunk": float(pct_chunk),
                            "status": filter_status,
                            "reason": filter_reason if filter_reason else "",
                        }
                    )

                status_with_reason = f"[{filter_status}{': ' + filter_reason if filter_reason else ''}]"
                logger.info(
                    f"    '{match['ngram']}' -> {entity_name} (score: {score:.3f}{stats_str}) {status_with_reason}"
                )

        # Step 4: Update cache
        cache_update_start = time.time()
        if self.enable_cache:
            self._update_cache(ngram_list, all_entities)
            cache_update_elapsed = time.time() - cache_update_start
            logger.info(f"  Cache update: {cache_update_elapsed:.3f}s")

        return all_entities, dict(ngram_to_entities)

    async def _compute_embeddings_batch(self, ngrams: list[str]) -> list:
        """
        Compute embeddings for n-grams using label encoder.

        Args:
            ngrams: List of n-gram strings

        Returns:
            List of embeddings (numpy arrays)
        """
        # Use label embedding model (MedCPT-Query-Encoder)
        embeddings = await self.embedding_manager.embed_labels(ngrams)
        return embeddings

    async def _search_entities(self, ngrams: list[str], embeddings: list) -> list[dict]:
        """
        Search entity collections with n-gram embeddings.

        MilvusStorage automatically:
        - Discovers collections starting with "entities_" (filtered by entity_collections if specified)
        - Searches each collection using label_embedding field in parallel
        - Returns combined results sorted by score

        Args:
            ngrams: List of n-gram strings (for logging only)
            embeddings: List of embedding vectors

        Returns:
            List of matched entity dicts with full metadata
        """
        entities = []

        try:
            # Search entity collections (filtered by entity_collections if specified)
            results = await self.milvus_storage.search_entities(
                query_embeddings=embeddings,
                top_k=1,  # Top 1 match per n-gram
                min_similarity=self.min_similarity,  # Pass threshold from ngram config
                include_collections=self.entity_collections,
            )

            # Extract full entity records from results
            for result_list in results:
                for result in result_list:
                    if "entity_name" in result:
                        entities.append(result)

        except Exception as e:
            logger.error(f"Entity search failed: {e}")

        return entities

    def _update_cache(self, ngrams: list[str], entities: list[dict]):
        """
        Update n-gram cache with results.

        Args:
            ngrams: List of n-grams that were matched
            entities: List of all matched entity dicts
        """
        # Build reverse mapping: which n-grams matched which entities
        # For simplicity, we'll cache at n-gram level
        # If any entity matched from this batch, consider n-grams positive

        if entities:
            # At least one match - add to positive cache
            for ngram in ngrams:
                if ngram not in self.positive_cache:
                    self.positive_cache[ngram] = []

                # Store all entities for this n-gram
                # (Simplified: store all entities from batch)
                self.positive_cache[ngram].extend(entities)

                # LRU eviction
                if len(self.positive_cache) > self.max_cache_size:
                    self.positive_cache.popitem(last=False)
        else:
            # No matches - add to negative cache
            for ngram in ngrams:
                self.negative_cache.add(ngram)

                # Limit negative cache size
                if len(self.negative_cache) > self.max_cache_size:
                    # Remove oldest (convert to list, remove first, convert back)
                    self.negative_cache = set(list(self.negative_cache)[1:])

    def _filter_by_statistics(self, entities: list[dict]) -> list[dict]:
        """
        Filter entities by pct_paper, pct_chunk, and num_text_variations thresholds.

        Rule: If entity NOT in statistics file, KEEP it (assumed high quality).

        Args:
            entities: List of entity dicts to filter

        Returns:
            Filtered list of entity dicts
        """
        if not self.entity_stats_manager:
            return entities  # No filtering

        if self.max_pct_paper is None and self.max_pct_chunk is None and self.max_num_text_variations is None:
            return entities  # No thresholds set

        # Extract entity names for filtering
        entity_names = [e.get("entity_name") for e in entities if e.get("entity_name")]

        # Get filtered entity names
        filtered_names = set(
            self.entity_stats_manager.filter_entities(
                entity_names, self.max_pct_paper, self.max_pct_chunk, self.max_num_text_variations
            )
        )

        # Return only entities whose names are in the filtered set
        return [e for e in entities if e.get("entity_name") in filtered_names]

    def clear_cache(self):
        """Clear all caches."""
        self.positive_cache.clear()
        self.negative_cache.clear()
        logger.info("N-gram matcher cache cleared")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache information
        """
        return {
            "positive_cache_size": len(self.positive_cache),
            "negative_cache_size": len(self.negative_cache),
            "max_cache_size": self.max_cache_size,
            "cache_enabled": self.enable_cache,
        }
