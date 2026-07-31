"""
Structured detailed logging for query pipeline.

Writes per-stage JSON/JSONL files for programmatic analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DetailedLogger:
    """
    Structured logger for query pipeline stages.

    Creates directory: {log_dir}/{query_id}/
    Writes JSON/JSONL files for each stage.

    Uses buffering for JSONL files to avoid file I/O overhead.
    """

    def __init__(self, log_dir: str | None, query_id: str | None):
        """
        Initialize detailed logger.

        Args:
            log_dir: Base directory for logs (None = disabled)
            query_id: Unique query identifier (None = disabled)
        """
        self.enabled = log_dir is not None and query_id is not None

        if self.enabled:
            self.query_dir = Path(log_dir) / query_id
            self.query_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Detailed logging enabled: {self.query_dir}")

            # Buffering for JSONL files (avoids opening/closing file 50K times)
            self._jsonl_buffers = {}  # filename -> list of lines to write
        else:
            self.query_dir = None
            self._jsonl_buffers = {}
            logger.debug("Detailed logging disabled")

    def write_json(self, filename: str, data: dict[str, Any]) -> None:
        """Write JSON file."""
        if not self.enabled:
            return

        file_path = self.query_dir / filename
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.debug(f"Wrote {filename}")

    def append_jsonl(self, filename: str, data: dict[str, Any]) -> None:
        """Append line to JSONL file (buffered)."""
        if not self.enabled:
            return

        # Add to buffer instead of writing immediately
        if filename not in self._jsonl_buffers:
            self._jsonl_buffers[filename] = []

        # Serialize to JSON string now (to catch errors early)
        json_line = json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)
        self._jsonl_buffers[filename].append(json_line)

    def flush(self) -> None:
        """Flush all buffered JSONL data to disk."""
        if not self.enabled:
            return

        for filename, lines in self._jsonl_buffers.items():
            if not lines:
                continue

            file_path = self.query_dir / filename
            with open(file_path, "w") as f:
                f.write("\n".join(lines))
                f.write("\n")

            logger.debug(f"Flushed {len(lines)} lines to {filename}")

        # Clear buffers after flushing
        self._jsonl_buffers.clear()

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        """Write metadata.json"""
        self.write_json("metadata.json", metadata)

    def log_entity_stage1_match(self, match_data: dict[str, Any]) -> None:
        """Append to entities_stage1.jsonl"""
        self.append_jsonl("entities_stage1.jsonl", match_data)

    def log_entity_stage2_candidate(self, candidate_data: dict[str, Any]) -> None:
        """Append to entities_stage2.jsonl"""
        self.append_jsonl("entities_stage2.jsonl", candidate_data)

    def log_entities_final(self, final_data: dict[str, Any]) -> None:
        """Write entities_final.json"""
        self.write_json("entities_final.json", final_data)

    def log_query_expansion(self, expansion_data: dict[str, Any]) -> None:
        """Write query_expansion.json"""
        self.write_json("query_expansion.json", expansion_data)

    def log_keyword_expansion(self, keyword_data: dict[str, Any]) -> None:
        """Write keyword_expansion.json"""
        self.write_json("keyword_expansion.json", keyword_data)

    def log_retrieval_kg_chunk(self, chunk_data: dict[str, Any]) -> None:
        """Append chunk to retrieval_kg.jsonl"""
        self.append_jsonl("retrieval_kg.jsonl", chunk_data)

    def log_retrieval_kg_summary(self, summary_data: dict[str, Any]) -> None:
        """Write retrieval_kg_summary.json"""
        self.write_json("retrieval_kg_summary.json", summary_data)

    def log_retrieval_elasticsearch_chunk(self, chunk_data: dict[str, Any]) -> None:
        """Append chunk to retrieval_elasticsearch.jsonl"""
        self.append_jsonl("retrieval_elasticsearch.jsonl", chunk_data)

    def log_retrieval_elasticsearch_summary(self, summary_data: dict[str, Any]) -> None:
        """Write retrieval_elasticsearch_summary.json"""
        self.write_json("retrieval_elasticsearch_summary.json", summary_data)

    def log_retrieval_milvus_chunk(self, chunk_data: dict[str, Any]) -> None:
        """Append chunk to retrieval_milvus.jsonl"""
        self.append_jsonl("retrieval_milvus.jsonl", chunk_data)

    def log_retrieval_milvus_summary(self, summary_data: dict[str, Any]) -> None:
        """Write retrieval_milvus_summary.json"""
        self.write_json("retrieval_milvus_summary.json", summary_data)

    def log_merge_statistics(self, merge_data: dict[str, Any]) -> None:
        """Write merge_statistics.json"""
        self.write_json("merge_statistics.json", merge_data)

    def log_rerank_stage1_chunk(self, chunk_data: dict[str, Any]) -> None:
        """Append chunk to rerank_stage1.jsonl"""
        self.append_jsonl("rerank_stage1.jsonl", chunk_data)

    def log_rerank_stage1_summary(self, summary_data: dict[str, Any]) -> None:
        """Write rerank_stage1_summary.json"""
        self.write_json("rerank_stage1_summary.json", summary_data)

    def log_rerank_stage2_chunk(self, chunk_data: dict[str, Any]) -> None:
        """Append chunk to rerank_stage2.jsonl"""
        self.append_jsonl("rerank_stage2.jsonl", chunk_data)

    def log_rerank_stage2_summary(self, summary_data: dict[str, Any]) -> None:
        """Write rerank_stage2_summary.json"""
        self.write_json("rerank_stage2_summary.json", summary_data)

    def log_final_top20(self, top20_data: dict[str, Any]) -> None:
        """Write final_top20.json"""
        self.write_json("final_top20.json", top20_data)

    def log_llm_response(self, llm_data: dict[str, Any]) -> None:
        """Write llm_response.json"""
        self.write_json("llm_response.json", llm_data)
