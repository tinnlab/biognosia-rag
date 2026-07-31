"""
Second-stage semantic community entity discovery.

Discovers related entities by treating Stage 1 entities as a semantic community
and performing ensemble-based similarity search with shuffled text combinations.
"""

import itertools
import logging
import math
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class SemanticCommunityDiscovery:
    """
    Discover related entities through bootstrap sampling and ensemble queries.

    Takes Stage 1 entities and their matching text snippets, generates shuffled
    combinations, and finds entities that consistently appear in semantic searches.
    """

    def __init__(
        self,
        embedding_manager,
        milvus_storage,
        num_shuffles: int = 1000,
        appearance_threshold: float = 0.5,
        min_similarity: float = 0.85,
        top_k: int = 10,
        min_entities: int = 2,
        embedding_batch_size: int = 200,
        query_batch_size: int = 100,
        shuffle_strategy: str = "adaptive",
        max_exhaustive_snippets: int = 6,
        min_unique_ratio: float = 0.8,
        exclude_collections: list[str] | None = None,
        rerank_processor=None,
        rerank_config: dict | None = None,
        enable_outlier_removal: bool = True,
        outlier_threshold: float = 0.60,
    ):
        """
        Initialize semantic community discovery.

        Args:
            embedding_manager: Embedding manager from src_rag
            milvus_storage: Milvus storage from src_rag
            num_shuffles: Target number of shuffled queries to generate
            appearance_threshold: Minimum appearance frequency (0.0-1.0)
            min_similarity: Minimum cosine similarity for entity matching
            top_k: Number of entities to retrieve per query
            min_entities: Minimum Stage 1 entities required to trigger Stage 2
            embedding_batch_size: Batch size for embedding generation
            query_batch_size: Batch size for Milvus entity search
            shuffle_strategy: "random", "exhaustive", or "adaptive"
            max_exhaustive_snippets: Max snippets for exhaustive permutations
            min_unique_ratio: Minimum unique shuffle ratio threshold
            exclude_collections: List of collection names to exclude from Stage 2 lookups
            rerank_processor: RerankProcessor for entity reranking (optional)
            rerank_config: Reranking configuration dict (optional)
            enable_outlier_removal: Whether to remove outlier snippets before shuffling
            outlier_threshold: Similarity threshold for outlier detection (0.0-1.0)
        """
        self.embedding_manager = embedding_manager
        self.milvus_storage = milvus_storage
        self.num_shuffles = num_shuffles
        self.appearance_threshold = appearance_threshold
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.min_entities = min_entities
        self.embedding_batch_size = embedding_batch_size
        self.query_batch_size = query_batch_size
        self.shuffle_strategy = shuffle_strategy
        self.max_exhaustive_snippets = max_exhaustive_snippets
        self.min_unique_ratio = min_unique_ratio
        self.exclude_collections = exclude_collections or []
        self.rerank_processor = rerank_processor
        self.rerank_config = rerank_config or {}
        self.enable_outlier_removal = enable_outlier_removal
        self.outlier_threshold = outlier_threshold

        # Log configuration
        if self.exclude_collections:
            logger.info(f"Stage 2 initialized with excluded collections: {', '.join(self.exclude_collections)}")
        else:
            logger.info("Stage 2 initialized: searching all collections")

    async def discover_entities(
        self,
        stage1_entities: list[dict],
        entity_text_snippets: dict[str, list[str]],
        detailed_logger=None,
    ) -> list[dict]:
        """
        Discover additional entities through semantic community analysis.

        Args:
            stage1_entities: Entities found in Stage 1
            entity_text_snippets: Mapping of entity_name -> list of text snippets
                                 that matched this entity in the query
            detailed_logger: Optional DetailedLogger for structured logging

        Returns:
            List of newly discovered entities (excludes Stage 1 entities)
        """
        # Check if we have enough entities to form a community
        if len(stage1_entities) < self.min_entities:
            logger.info(
                f"Stage 2 skipped: only {len(stage1_entities)} entities (minimum required: {self.min_entities})"
            )
            return []

        # Generate shuffled queries
        logger.info(f"Stage 2: Generating {self.num_shuffles} shuffled queries from text snippets...")
        shuffled_queries = await self._generate_shuffled_queries(entity_text_snippets)

        if not shuffled_queries:
            logger.warning("Stage 2: No shuffled queries generated")
            return []

        # Log shuffled queries BEFORE querying (so they appear once, not repeated per entity)
        logger.info(f"Stage 2: Generated {len(shuffled_queries)} shuffled queries for entity discovery")

        # Log first 3 sample queries at INFO level
        num_samples = min(3, len(shuffled_queries))
        for i in range(num_samples):
            query_preview = shuffled_queries[i][:150] + "..." if len(shuffled_queries[i]) > 150 else shuffled_queries[i]
            logger.info(f"  Sample query {i + 1}: {query_preview}")

        if len(shuffled_queries) > num_samples:
            logger.info(f"  ... and {len(shuffled_queries) - num_samples} more queries")

        # Query entity collections
        # Log which collections will be searched
        if self.exclude_collections:
            logger.info(
                f"Stage 2: Querying entity collections with {len(shuffled_queries)} queries "
                f"(excluding: {', '.join(self.exclude_collections)}) using content embeddings (vector field)..."
            )
        else:
            logger.info(
                f"Stage 2: Querying entity collections with {len(shuffled_queries)} queries "
                f"(all collections) using content embeddings (vector field)..."
            )

        all_results = await self._query_entities_batch(shuffled_queries)

        # Log results from top 5 queries for debugging
        if logger.isEnabledFor(logging.DEBUG) and all_results:
            num_to_show = min(5, len(all_results))
            logger.debug(f"Stage 2: Results from top {num_to_show} queries:")
            for i, result_list in enumerate(all_results[:num_to_show], 1):
                if result_list:
                    entity_names = [e.get("entity_name", "Unknown") for e in result_list[:5]]
                    logger.debug(f"  Query {i}: Found {len(result_list)} entities, top 5: {', '.join(entity_names)}")
                else:
                    logger.debug(f"  Query {i}: No entities found")

        # Count entity appearances and track which queries matched each entity
        # Also track similarity scores for statistics
        entity_appearances = defaultdict(int)
        entity_records = {}
        entity_to_queries = defaultdict(list)
        entity_similarities = defaultdict(list)  # Track all similarity scores per entity
        entity_best_query = {}  # Track best matching shuffled query for each entity

        for query_idx, result_list in enumerate(all_results):
            for entity in result_list:
                entity_name = entity.get("entity_name")
                if entity_name:
                    entity_appearances[entity_name] += 1
                    entity_to_queries[entity_name].append(query_idx)

                    # Track similarity score
                    score = entity.get("score", 0.0)
                    entity_similarities[entity_name].append(score)

                    # Track best matching query (highest similarity score)
                    if entity_name not in entity_best_query or score > entity_best_query[entity_name]["score"]:
                        entity_best_query[entity_name] = {
                            "query_idx": query_idx,
                            "score": score,
                            "query_text": shuffled_queries[query_idx],
                        }

                    if entity_name not in entity_records:
                        entity_records[entity_name] = entity

        # Apply reranking to ALL candidates BEFORE filtering (so scores are in entity_records)
        stage1_entity_names = {e.get("entity_name") for e in stage1_entities if e.get("entity_name")}

        # Build list of ALL candidate entities (excluding Stage 1)
        all_candidate_entities = []
        for entity_name, count in entity_appearances.items():
            if entity_name not in stage1_entity_names:
                entity_record = entity_records[entity_name].copy()
                entity_record["stage2_appearance_count"] = count
                entity_record["stage2_appearance_percentage"] = count / len(shuffled_queries)

                # Attach best matching shuffled query for reranking
                if entity_name in entity_best_query:
                    entity_record["stage2_best_query"] = entity_best_query[entity_name]["query_text"]
                    entity_record["stage2_best_query_score"] = entity_best_query[entity_name]["score"]

                all_candidate_entities.append(entity_record)

        # Rerank ALL candidates if enabled
        if self.rerank_processor and self.rerank_config.get("enable_stage2", False):
            await self._apply_reranking(all_candidate_entities)
            # Update entity_records with rerank scores
            for entity in all_candidate_entities:
                entity_name = entity.get("entity_name")
                if entity_name and "entity_rerank_score" in entity:
                    entity_records[entity_name]["entity_rerank_score"] = entity["entity_rerank_score"]
        else:
            logger.debug("Stage 2 entity reranking: disabled or no rerank processor")

        # Filter by frequency threshold
        threshold_count = int(self.appearance_threshold * len(shuffled_queries))
        discovered_entities = [e for e in all_candidate_entities if e["stage2_appearance_count"] >= threshold_count]
        discovered_entities.sort(key=lambda x: x["stage2_appearance_count"], reverse=True)

        logger.info(
            f"Stage 2: Discovered {len(discovered_entities)} new entities from {len(entity_appearances)} candidates"
        )

        # Log ALL candidates with similarity statistics (sorted by appearance count)
        threshold_count = int(self.appearance_threshold * len(shuffled_queries))
        logger.info(
            f"Stage 2: All candidates (threshold: {threshold_count}/{len(shuffled_queries)} "
            f"= {self.appearance_threshold:.0%}):"
        )

        # Sort all candidates by appearance count
        all_candidates = sorted(entity_appearances.items(), key=lambda x: x[1], reverse=True)

        for entity_name, count in all_candidates:
            # Skip Stage 1 entities
            if entity_name in stage1_entity_names:
                continue

            percentage = count / len(shuffled_queries)
            similarities = entity_similarities.get(entity_name, [])

            # Compute similarity statistics
            if similarities:
                min_sim = min(similarities)
                max_sim = max(similarities)
                mean_sim = sum(similarities) / len(similarities)
            else:
                min_sim = max_sim = mean_sim = 0.0

            # Mark if discovered
            status = "DISCOVERED" if count >= threshold_count else "filtered"

            # Get rerank score if available (will be None before reranking)
            entity_record = entity_records.get(entity_name)
            rerank_score = entity_record.get("entity_rerank_score") if entity_record else None

            # Build log message
            log_msg = (
                f"  {entity_name}: {count}/{len(shuffled_queries)} queries ({percentage:.1%}) "
                f"[{status}] (sim: min={min_sim:.3f}, max={max_sim:.3f}, mean={mean_sim:.3f}"
            )

            # Add rerank score if available
            if rerank_score is not None:
                log_msg += f", rerank={rerank_score:.4f}"

            log_msg += ")"
            logger.info(log_msg)

            # Detailed logging: log Stage 2 candidate
            if detailed_logger:
                detailed_logger.log_entity_stage2_candidate(
                    {
                        "entity_id": entity_name,
                        "query_matches": count,
                        "total_queries": len(shuffled_queries),
                        "pct": float(percentage),
                        "sim_mean": float(mean_sim),
                        "rerank_score": float(rerank_score) if rerank_score is not None else None,
                        "status": status,
                    }
                )

        return discovered_entities

    async def _apply_reranking(self, entities: list[dict]) -> list[dict]:
        """
        Apply reranking to Stage 2 entities using their best matching queries.

        Updates entities IN-PLACE with entity_rerank_score field.

        Args:
            entities: Stage 2 discovered entities with stage2_best_query field

        Returns:
            Same entities list (updated in-place)
        """
        from ..rerank.processor import rerank_entities

        if not entities:
            return entities

        logger.info(f"Stage 2 reranking: reranking {len(entities)} entities...")

        # Call rerank_entities with query (won't be used for Stage 2)
        # Stage 2 entities will use their stage2_best_query field
        await rerank_entities(
            query="",  # Placeholder, not used for Stage 2
            entities=entities,  # Updated IN-PLACE
            rerank_config=self.rerank_config,
            rerank_processor=self.rerank_processor,
            stage="stage2",
        )

        # Count how many have scores
        scored_count = sum(1 for e in entities if "entity_rerank_score" in e)
        logger.info(f"Stage 2 reranking: added scores to {scored_count}/{len(entities)} entities")

        return entities

    async def _remove_outlier_snippets(
        self,
        category_snippets: dict[str, set[str]],
        entity_text_snippets: dict[str, list[str]],
    ) -> dict[str, set[str]]:
        """
        Remove outlier snippets using entity centroid similarity.

        Filters out semantically unrelated snippets (e.g., instruction phrases,
        generic terms) that would contaminate shuffled queries.

        Method:
        1. For each category, extract all entity IDs
        2. Embed entity IDs using stage2_model (bge-m3)
        3. Compute entity centroid
        4. Embed snippet texts and measure similarity to entity centroid
        5. Remove snippets with similarity < outlier_threshold

        Args:
            category_snippets: Dict mapping category to set of snippet texts
            entity_text_snippets: Original entity->snippets mapping for extracting entity IDs

        Returns:
            Cleaned category_snippets with outliers removed
        """
        import numpy as np

        if not self.enable_outlier_removal:
            logger.info("Stage 2: Outlier removal disabled")
            return category_snippets

        threshold = self.outlier_threshold
        logger.info(f"Stage 2: Removing outlier snippets (threshold={threshold:.2f})")

        cleaned_category_snippets = {}
        total_removed = 0

        for category, snippets in category_snippets.items():
            logger.info(f"  [{category}] Processing {len(snippets)} snippets")

            if len(snippets) < 3:
                cleaned_category_snippets[category] = snippets
                continue

            category_entities = set()
            sample_entity_names = []
            for entity_name in entity_text_snippets.keys():
                if ":" in entity_name:
                    entity_category = entity_name.split(":", 1)[0]
                    if entity_category == category:
                        entity_id = entity_name.split(":", 1)[1]
                        category_entities.add(entity_id)
                        if len(sample_entity_names) < 3:
                            sample_entity_names.append(f"{entity_name} -> {entity_id}")

            logger.info(f"  [{category}] Found {len(category_entities)} entities from {len(snippets)} snippets")
            logger.info(f"  [{category}] Entity extraction: {sample_entity_names}")

            if len(category_entities) < 3:
                logger.info(f"  [{category}] Skipping outlier removal (need >=3 entities)")
                cleaned_category_snippets[category] = snippets
                continue

            try:
                entity_list = list(category_entities)
                snippet_list = list(snippets)

                logger.info(f"  [{category}] Embedding {len(entity_list)} entity IDs with CLS pooling...")
                entity_embeddings = await self.embedding_manager.embed_for_outlier_detection(entity_list)

                if entity_embeddings.size == 0 or len(entity_embeddings.shape) == 0:
                    logger.warning(f"  [{category}] Empty embeddings for entities, skipping outlier removal")
                    cleaned_category_snippets[category] = snippets
                    continue

                entity_centroid = np.mean(entity_embeddings, axis=0, keepdims=True)
                entity_centroid = entity_centroid / np.linalg.norm(entity_centroid)

                logger.info(f"  [{category}] Embedding {len(snippet_list)} snippet texts with CLS pooling...")
                snippet_embeddings = await self.embedding_manager.embed_for_outlier_detection(snippet_list)

                if snippet_embeddings.size == 0 or len(snippet_embeddings.shape) == 0:
                    logger.warning(f"  [{category}] Empty embeddings for snippets, skipping outlier removal")
                    cleaned_category_snippets[category] = snippets
                    continue

                logger.info(f"  [{category}] Computing similarities (threshold={threshold})...")
                similarities = (snippet_embeddings @ entity_centroid.T).flatten()
                sim_range = f"[{similarities.min():.3f}, {similarities.max():.3f}]"
                logger.info(f"  [{category}] Computed {len(similarities)} similarities, range: {sim_range}")

                kept_snippets = set()
                outliers = []

                for snippet, sim in zip(snippet_list, similarities):
                    if sim >= threshold:
                        kept_snippets.add(snippet)
                    else:
                        outliers.append((snippet, sim))

                cleaned_category_snippets[category] = kept_snippets
                removed_count = len(snippets) - len(kept_snippets)
                total_removed += removed_count

                logger.info(f"  [{category}] Filtered: {removed_count} removed, {len(kept_snippets)} kept")

                if removed_count > 0:
                    logger.info(
                        f"  [{category}] Removed {removed_count}/{len(snippets)} outlier snippets "
                        f"(kept {len(kept_snippets)})"
                    )
                    if outliers and logger.isEnabledFor(logging.DEBUG):
                        for snippet, sim in sorted(outliers, key=lambda x: x[1])[:5]:
                            logger.debug(f"    Outlier: '{snippet}' (sim={sim:.3f})")
            except Exception as e:
                logger.error(f"  [{category}] Error during outlier detection: {e}", exc_info=True)
                cleaned_category_snippets[category] = snippets
                continue

        logger.info(f"Stage 2: Outlier removal complete. Removed {total_removed} snippets total")

        return cleaned_category_snippets

    async def _generate_shuffled_queries(
        self,
        entity_text_snippets: dict[str, list[str]],
    ) -> list[str]:
        """
        Generate unique shuffled queries using adaptive strategy.

        CRITICAL: Groups entities by category (prefix like GENE:, GO:, KEGG:)
        and generates separate shuffled queries for each category to maintain
        semantic coherence.

        Strategies:
        - exhaustive: Generate all N! permutations (guaranteed unique)
        - random: Random shuffling (fast, may duplicate)
        - adaptive: Exhaustive if small, random if large
        """
        # Group entities by category (prefix before ':')
        from collections import defaultdict

        category_snippets = defaultdict(set)

        for entity_name, snippets in entity_text_snippets.items():
            # Extract category from entity_name (e.g., "GENE:BRCA1" -> "GENE")
            if ":" in entity_name:
                category = entity_name.split(":", 1)[0]
            else:
                category = "OTHER"

            # Add snippets to this category
            category_snippets[category].update(snippets)

        # Log category distribution
        category_counts = {cat: len(snippets) for cat, snippets in category_snippets.items()}
        logger.info(f"Stage 2: Grouped entities into {len(category_snippets)} categories: {category_counts}")

        # Remove outlier snippets before shuffle generation
        category_snippets = await self._remove_outlier_snippets(category_snippets, entity_text_snippets)

        # Generate shuffled queries for each category separately
        all_shuffled_queries = []

        # First pass: Filter categories with sufficient diversity (>= 3 snippets)
        # and calculate total snippets for proportional distribution
        valid_categories = {
            category: snippet_set for category, snippet_set in category_snippets.items() if len(snippet_set) >= 3
        }

        if not valid_categories:
            logger.warning("No categories with sufficient diversity (>= 3 snippets) for Stage 2")
            return []

        # Log skipped categories
        skipped_categories = set(category_snippets.keys()) - set(valid_categories.keys())
        if skipped_categories:
            for category in skipped_categories:
                num_snippets = len(category_snippets[category])
                if num_snippets == 0:
                    logger.warning(f"Category {category}: Skipping (no snippets)")
                elif num_snippets < 3:
                    logger.info(
                        f"Category {category}: Skipping ({num_snippets} snippets, "
                        f"minimum 3 required for community discovery)"
                    )

        # Calculate total snippets from valid categories only
        total_snippets = sum(len(s) for s in valid_categories.values())

        for category, snippet_set in valid_categories.items():
            all_snippets = list(snippet_set)
            num_snippets = len(all_snippets)

            # Distribute num_shuffles proportionally based on number of snippets
            # (categories with more entities get more shuffles)
            category_shuffles = max(1, int(self.num_shuffles * num_snippets / total_snippets))

            logger.info(f"Category {category}: Generating {category_shuffles} queries from {num_snippets} snippets")

            # Determine strategy
            if self.shuffle_strategy == "exhaustive":
                use_exhaustive = True
            elif self.shuffle_strategy == "random":
                use_exhaustive = False
            else:  # adaptive
                use_exhaustive = num_snippets <= self.max_exhaustive_snippets

            shuffled_queries = set()

            if use_exhaustive:
                # Generate ALL unique permutations
                factorial = math.factorial(num_snippets)
                logger.info(f"  Using exhaustive permutations ({num_snippets}! = {factorial:,} permutations)")

                for perm in itertools.permutations(all_snippets):
                    combined = " ".join(perm)
                    shuffled_queries.add(combined)

                    # Stop if we've reached target
                    if len(shuffled_queries) >= category_shuffles:
                        break
            else:
                # Random shuffling with deduplication
                logger.info(f"  Using random shuffling ({num_snippets} snippets, target: {category_shuffles} shuffles)")

                # Try up to 10x target to get enough unique shuffles
                max_attempts = category_shuffles * 10
                attempts = 0

                while len(shuffled_queries) < category_shuffles and attempts < max_attempts:
                    shuffled = all_snippets.copy()
                    random.shuffle(shuffled)

                    combined = " ".join(shuffled)
                    shuffled_queries.add(combined)
                    attempts += 1

            # Check uniqueness ratio
            unique_ratio = len(shuffled_queries) / category_shuffles if category_shuffles > 0 else 0
            if unique_ratio < self.min_unique_ratio:
                logger.warning(
                    f"  Category {category}: Low shuffle diversity: {len(shuffled_queries)} unique shuffles "
                    f"from {category_shuffles} target ({unique_ratio:.1%}). "
                    f"Consider using exhaustive strategy or lowering num_shuffles."
                )

            logger.info(f"  Category {category}: Generated {len(shuffled_queries)} unique shuffled queries")
            all_shuffled_queries.extend(shuffled_queries)

        logger.info(f"Generated {len(all_shuffled_queries)} total unique shuffled queries across all categories")
        return list(all_shuffled_queries)

    async def _query_entities_batch(
        self,
        shuffled_queries: list[str],
    ) -> list[list[dict]]:
        """
        Query entities with two-level batching.

        Level 1 (embedding_batch_size): Batch embedding computation
        Level 2 (query_batch_size): Batch Milvus queries
        """
        all_results = []

        # Level 1: Embedding batching (GPU optimization)
        for emb_start in range(0, len(shuffled_queries), self.embedding_batch_size):
            emb_end = min(emb_start + self.embedding_batch_size, len(shuffled_queries))
            emb_batch = shuffled_queries[emb_start:emb_end]

            # Batch embed using Stage 2 model (bge-m3 with 2048 max_length) for longer text
            # Stage 2 queries are long text snippets, not short labels
            embeddings = await self.embedding_manager.embed_stage2(emb_batch)

            # Level 2: Milvus query batching (network optimization)
            for query_start in range(0, len(embeddings), self.query_batch_size):
                query_end = min(query_start + self.query_batch_size, len(embeddings))
                query_embeddings = embeddings[query_start:query_end]

                # Batch search with collection filtering
                # Use vector field (content embeddings) for Stage 2 long text queries
                results = await self.milvus_storage.search_entities(
                    query_embeddings=query_embeddings,
                    top_k=self.top_k,
                    min_similarity=self.min_similarity,
                    exclude_collections=self.exclude_collections,
                    search_field="vector",  # Use content embeddings for long text
                )

                all_results.extend(results)

        return all_results

    def _filter_by_frequency(
        self,
        entity_appearances: dict[str, int],
        entity_records: dict[str, dict],
        total_queries: int,
        stage1_entity_names: set[str],
        entity_best_query: dict[str, dict],
    ) -> list[dict]:
        """
        Filter entities by appearance frequency.

        Keep entities that:
        - Appear in >= threshold * total_queries
        - Are NOT in Stage 1 entities (avoid duplicates)
        """
        threshold_count = int(self.appearance_threshold * total_queries)

        discovered_entities = []
        for entity_name, count in entity_appearances.items():
            # Skip Stage 1 entities (already included)
            if entity_name in stage1_entity_names:
                continue

            # Check threshold
            if count >= threshold_count:
                entity_record = entity_records[entity_name].copy()
                entity_record["stage2_appearance_count"] = count
                entity_record["stage2_appearance_percentage"] = count / total_queries

                # Attach best matching shuffled query for reranking
                if entity_name in entity_best_query:
                    entity_record["stage2_best_query"] = entity_best_query[entity_name]["query_text"]
                    entity_record["stage2_best_query_score"] = entity_best_query[entity_name]["score"]

                discovered_entities.append(entity_record)

        # Sort by appearance count (descending)
        discovered_entities.sort(key=lambda x: x["stage2_appearance_count"], reverse=True)

        return discovered_entities
