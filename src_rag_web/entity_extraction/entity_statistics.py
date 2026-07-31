"""
Entity statistics management for quality filtering.

Loads entity statistics from TSV/CSV files and provides filtering based on
pct_paper and pct_chunk thresholds.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EntityStatisticsManager:
    """Manage entity statistics for quality filtering."""

    def __init__(self, stats_file_path: str | None = None):
        """
        Initialize entity statistics manager.

        Args:
            stats_file_path: Path to entity statistics file (TSV or CSV)
        """
        self.stats_file_path = stats_file_path
        self.statistics = {}  # entity_name -> {pct_paper, pct_chunk}
        self.loaded = False

    def load_statistics(self):
        """
        Load entity statistics from TSV/CSV file into memory.

        Expected columns (will extract these):
        - entity_name: Entity identifier (e.g., GENE:BRCA1)
        - pct_chunk: Percentage of chunks entity appears in
        - pct_paper: Percentage of papers entity appears in
        - all_extracted_text: Pipe-separated list of text variations (optional)

        Supports both TSV and CSV formats (auto-detected by file extension).
        """
        if not self.stats_file_path:
            logger.warning("No entity statistics file specified")
            return

        stats_file = Path(self.stats_file_path)
        if not stats_file.exists():
            logger.warning(f"Entity statistics file not found: {stats_file}")
            return

        logger.info(f"Loading entity statistics from {stats_file}...")

        # Determine delimiter based on file extension
        delimiter = "\t" if stats_file.suffix.lower() in [".tsv", ".txt"] else ","

        try:
            import pandas as pd

            df = pd.read_csv(stats_file, sep=delimiter)

            # Validate required columns
            required_columns = {"entity_name", "pct_paper", "pct_chunk"}
            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"Statistics file missing required columns. Expected: {required_columns}, Found: {set(df.columns)}"
                )

            # Check if all_extracted_text column exists
            has_text_variations = "all_extracted_text" in df.columns

            # Extract columns
            for _, row in df.iterrows():
                entity_name = row["entity_name"]
                stats_dict = {
                    "pct_paper": float(row["pct_paper"]),
                    "pct_chunk": float(row["pct_chunk"]),
                }

                # Count text variations if available
                if has_text_variations:
                    all_extracted_text = row.get("all_extracted_text", "")
                    if pd.notna(all_extracted_text) and all_extracted_text:
                        # Split by '|' and count unique non-empty texts
                        text_variations = [t.strip() for t in str(all_extracted_text).split("|") if t.strip()]
                        stats_dict["num_text_variations"] = len(text_variations)
                    else:
                        stats_dict["num_text_variations"] = 0
                else:
                    stats_dict["num_text_variations"] = None  # Column not available

                self.statistics[entity_name] = stats_dict

            self.loaded = True
            text_info = "with text variations" if has_text_variations else "without text variations"
            logger.info(f"Loaded statistics for {len(self.statistics)} entities ({text_info})")

        except Exception as e:
            logger.error(f"Failed to load entity statistics: {e}")
            raise

    def get_statistics(self, entity_name: str) -> dict | None:
        """
        Get statistics for a specific entity.

        Args:
            entity_name: Entity name to look up

        Returns:
            Dict with pct_paper and pct_chunk, or None if not found
        """
        return self.statistics.get(entity_name)

    def get_filter_reason(
        self,
        entity_name: str,
        max_pct_paper: float | None = None,
        max_pct_chunk: float | None = None,
        max_num_text_variations: int | None = None,
    ) -> str | None:
        """
        Get reason why entity would be filtered out.

        Args:
            entity_name: Entity name to check
            max_pct_paper: Maximum percentage of papers
            max_pct_chunk: Maximum percentage of chunks
            max_num_text_variations: Maximum number of text variations

        Returns:
            Reason string if entity should be filtered, None if it passes
        """
        # Entity not in statistics -> KEEP (high quality)
        if entity_name not in self.statistics:
            return None

        stats = self.statistics[entity_name]

        # Check pct_paper
        if max_pct_paper is not None:
            if stats["pct_paper"] >= max_pct_paper:
                return f"pct_paper={stats['pct_paper']:.1f}%>={max_pct_paper}%"

        # Check pct_chunk
        if max_pct_chunk is not None:
            if stats["pct_chunk"] >= max_pct_chunk:
                return f"pct_chunk={stats['pct_chunk']:.1f}%>={max_pct_chunk}%"

        # Check num_text_variations
        if max_num_text_variations is not None:
            num_texts = stats.get("num_text_variations")
            if num_texts is not None and num_texts >= max_num_text_variations:
                return f"num_texts={num_texts}>={max_num_text_variations}"

        return None  # Passes all filters

    def is_high_quality(
        self,
        entity_name: str,
        max_pct_paper: float | None = None,
        max_pct_chunk: float | None = None,
        max_num_text_variations: int | None = None,
    ) -> bool:
        """
        Check if entity passes quality thresholds.

        Rule: If entity NOT in statistics file, it's considered HIGH QUALITY
        (assumed to be rare/novel entity).

        Args:
            entity_name: Entity name to check
            max_pct_paper: Maximum percentage of papers (filter if >=)
            max_pct_chunk: Maximum percentage of chunks (filter if >=)
            max_num_text_variations: Maximum number of text variations (filter if >=)

        Returns:
            True if entity is high quality, False if should be filtered out
        """
        # Entity not in statistics -> KEEP (high quality)
        if entity_name not in self.statistics:
            return True

        stats = self.statistics[entity_name]

        # Filter by pct_paper
        if max_pct_paper is not None:
            if stats["pct_paper"] >= max_pct_paper:
                return False  # Too common in papers

        # Filter by pct_chunk
        if max_pct_chunk is not None:
            if stats["pct_chunk"] >= max_pct_chunk:
                return False  # Too common in chunks

        # Filter by num_text_variations
        if max_num_text_variations is not None:
            num_texts = stats.get("num_text_variations")
            if num_texts is not None and num_texts >= max_num_text_variations:
                return False  # Too many text variations (too generic)

        return True

    def filter_entities(
        self,
        entity_names: list,
        max_pct_paper: float | None = None,
        max_pct_chunk: float | None = None,
        max_num_text_variations: int | None = None,
    ) -> list:
        """
        Filter a list of entity names by quality thresholds.

        Args:
            entity_names: List of entity names to filter
            max_pct_paper: Maximum percentage of papers (filter if >=)
            max_pct_chunk: Maximum percentage of chunks (filter if >=)
            max_num_text_variations: Maximum number of text variations (filter if >=)

        Returns:
            Filtered list of high-quality entity names
        """
        if not self.loaded or (max_pct_paper is None and max_pct_chunk is None and max_num_text_variations is None):
            return entity_names  # No filtering

        return [
            entity_name
            for entity_name in entity_names
            if self.is_high_quality(entity_name, max_pct_paper, max_pct_chunk, max_num_text_variations)
        ]

    def get_stats_summary(self) -> dict:
        """
        Get summary statistics about loaded data.

        Returns:
            Dict with summary information
        """
        if not self.loaded:
            return {"loaded": False, "total_entities": 0}

        return {"loaded": True, "total_entities": len(self.statistics), "file_path": str(self.stats_file_path)}
