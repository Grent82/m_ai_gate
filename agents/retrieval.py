from typing import Dict, List
from datetime import datetime
import heapq
from core.logger import setup_logger
from memory.long_term_memory import LongTermMemory, MemoryNode
from memory.short_term_memory import ShortTermMemory
from models.embeddings import get_embedding

logger = setup_logger(__name__, log_level="DEBUG")

class Retrieval:
    def __init__(self, long_term_memory: LongTermMemory, short_term_memory: ShortTermMemory):
        self.long_term_memory = long_term_memory
        self.short_term_memory = short_term_memory

    def retrieve_context(self, perceived_nodes: List[MemoryNode]) -> Dict[str, Dict[str, List[MemoryNode]]]:
        logger.info("[Retrieval] Starting contextual retrieval...")

        result = {}

        for node in perceived_nodes:
            description = node.event.description if node.event else "unknown"
            logger.debug(f"[Retrieval] Retrieving context for event: {description}")

            related_nodes = self.long_term_memory.retrieve_relevant_events(node.event)
            logger.debug(f"[Retrieval] Retrieved {len(related_nodes)} relevant events.")

            related_thoughts = self.long_term_memory.retrieve_relevant_thoughts(node.event)
            logger.debug(f"[Retrieval] Retrieved {len(related_thoughts)} relevant thoughts.")

            combined_nodes = list({n.node_id: n for n in related_nodes + related_thoughts}.values())
            logger.debug(f"[Retrieval] Combined total unique context nodes: {len(combined_nodes)}")

            result[description] = {
                "current_event": node.event,
                "context_nodes": combined_nodes
            }

        return result

    def retrieve_relevant_nodes(self, focal_descriptions: List[str]) -> Dict[str, List[MemoryNode]]:
        logger.info("[Retrieval] Starting vector-based relevance scoring...")

        max_results = 30
        result = {}

        all_nodes = [
            node for node_type in ["event", "thought"]
            for node in self.long_term_memory.node_sequences[node_type]
            if not node.is_expired()
        ]
        logger.debug(f"[Retrieval] Total valid nodes before filtering: {len(all_nodes)}")

        all_nodes = [node for node in all_nodes if self._is_non_idle(node)]
        logger.debug(f"[Retrieval] Nodes after idle-filtering: {len(all_nodes)}")

        all_nodes.sort(key=lambda n: n.created, reverse=True)

        for description in focal_descriptions:
            logger.debug(f"[Retrieval] Processing focal description: {description}")
            focus_emb = get_embedding(description)

            recency = self._extract_recency(all_nodes)
            importance = self._extract_importance(all_nodes)
            relevance = self._extract_relevance(all_nodes, focus_emb)

            logger.debug(f"[Retrieval] Raw recency: {recency}")
            logger.debug(f"[Retrieval] Raw importance: {importance}")
            logger.debug(f"[Retrieval] Raw relevance: {relevance}")

            recency = self._normalize_dict(recency)
            importance = self._normalize_dict(importance)
            relevance = self._normalize_dict(relevance)

            all_ids = set().union(recency.keys(), importance.keys(), relevance.keys())
            combined_score = {
                node_id: (
                    self.short_term_memory.recency_weight * recency.get(node_id, 0.0)
                    + self.short_term_memory.importance_weight * importance.get(node_id, 0.0)
                    + self.short_term_memory.relevance_weight * relevance.get(node_id, 0.0)
                )
                for node_id in all_ids
            }
            logger.debug(f"[Retrieval] Combined weighted scores: {combined_score}")

            top_ids = self._top_n(combined_score, max_results)
            ts = self.short_term_memory.current_time or datetime.now()
            top_nodes = [n for n in all_nodes if n.node_id in top_ids]
            for n in top_nodes:
                n.last_accessed = ts

            result[description] = top_nodes
            logger.debug(f"[Retrieval] Top {max_results} nodes for description '{description}': {top_ids}")

        return result

    def _is_non_idle(self, node: MemoryNode) -> bool:
        result = not node.event or "idle" not in node.event.description.lower()
        logger.debug(f"[Retrieval] Node '{node.node_id}' is_non_idle = {result}")
        return result

    def _top_n(self, d: Dict[str, float], n: int) -> List[str]:
        if not d or n <= 0:
            return []
        top_items = heapq.nlargest(n, d.items(), key=lambda item: item[1])
        top_keys = [k for k, _ in top_items]
        logger.debug(f"[Retrieval] Top {n} keys by score: {top_keys}")
        return top_keys

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
        logger.debug(f"[Retrieval] Cosine similarity: {similarity}")
        return similarity

    def _normalize_dict(self, values: Dict[str, float], target_min: float = 0.0, target_max: float = 1.0) -> Dict[str, float]:
        if not values:
            return {}
        min_val, max_val = min(values.values()), max(values.values())
        if max_val == min_val:
            midpoint = (target_max + target_min) / 2.0
            normalized = {k: midpoint for k in values.keys()}
            logger.debug(f"[Retrieval] Normalized values (flat range -> midpoint): {normalized}")
            return normalized
        range_val = max_val - min_val
        normalized = {
            k: (v - min_val) / range_val * (target_max - target_min) + target_min
            for k, v in values.items()
        }
        logger.debug(f"[Retrieval] Normalized values: {normalized}")
        return normalized

    def _extract_recency(self, nodes: List[MemoryNode]) -> Dict[str, float]:
        decay = self.short_term_memory.recency_decay
        recency = {
            node.node_id: decay ** idx
            for idx, node in enumerate(nodes)
        }
        logger.debug(f"[Retrieval] Extracted recency scores: {recency}")
        return recency

    def _extract_importance(self, nodes: List[MemoryNode]) -> Dict[str, float]:
        importance = {node.node_id: node.relevance for node in nodes}
        logger.debug(f"[Retrieval] Extracted importance scores: {importance}")
        return importance

    def _extract_relevance(self, nodes: List[MemoryNode], focus_embedding: List[float]) -> Dict[str, float]:
        relevance = {
            node.node_id: self._cosine_similarity(node.embedding, focus_embedding)
            for node in nodes if node.embedding is not None
        }
        logger.debug(f"[Retrieval] Extracted relevance scores: {relevance}")
        return relevance
