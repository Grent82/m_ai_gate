import re
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader

from core.agent import Agent
from core.agent_action import Event
from core.logger import setup_logger
from memory.long_term_memory import MemoryNode
from models.embeddings import get_embedding
from models.local_model import LocalModel

from .event_triple_generator import EventTripleGenerator
from .retrieval import Retrieval


logger = setup_logger(__name__)


class Reflection:
    """Generates reflective thoughts based on an agent's memories."""

    def __init__(
        self,
        model: LocalModel,
        retrieval: Retrieval,
        prompt_dir: str = "prompts",
    ) -> None:
        self.model = model
        self.retrieval = retrieval
        self.env = Environment(loader=FileSystemLoader(prompt_dir))
        self.event_generator = EventTripleGenerator(model, prompt_dir)
        # Default reset value for the importance trigger
        self.importance_threshold = 150.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reflect(self, agent: Agent) -> List[MemoryNode]:
        """Run the full reflection pipeline if the trigger is met."""
        if not self._should_reflect(agent):
            logger.debug("[Reflection] Trigger conditions not met.")
            return []

        logger.info("[Reflection] Starting reflection...")
        focal_points = self._generate_focal_points(agent)
        retrieved = self.retrieval.retrieve_relevant_nodes(focal_points)

        created_nodes: List[MemoryNode] = []

        for _, nodes in retrieved.items():
            insights = self._generate_insights_and_evidence(agent, nodes)
            for thought, evidence_ids in insights.items():
                event = self._thought_to_event(agent, thought)
                embedding = get_embedding(thought)
                keywords = {k for k in [event.subject, event.predicate, event.object] if k}
                relevance = self._thought_significance(agent, thought)
                node = agent.long_term_memory.add_thought(
                    event,
                    relevance=relevance,
                    keywords=keywords,
                    filling=evidence_ids,
                    embedding=embedding,
                )
                created_nodes.append(node)

        self._reset_reflection_counter(agent)
        logger.info(
            f"[Reflection] Stored {len(created_nodes)} reflective thoughts."  # noqa: E501
        )
        return created_nodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _should_reflect(self, agent: Agent) -> bool:
        stm = agent.short_term_memory
        logger.debug(
            f"[Reflection] importance_trigger_current={stm.importance_trigger_current}"
        )
        return (
            stm.importance_trigger_current <= 0
            and (
                agent.long_term_memory.node_sequences["event"]
                or agent.long_term_memory.node_sequences["thought"]
            )
        )

    def _generate_focal_points(self, agent: Agent, n: int = 3) -> List[str]:
        nodes = [
            node
            for node in (
                agent.long_term_memory.node_sequences["event"]
                + agent.long_term_memory.node_sequences["thought"]
            )
            if node.event and "idle" not in node.event.description.lower()
        ]
        nodes.sort(key=lambda n: n.last_accessed, reverse=True)

        count = agent.short_term_memory.importance_element_count or n
        selected = nodes[:count]
        statements = "\n".join(node.event.description for node in selected)

        template = self.env.get_template("generate_focal_points.txt")
        prompt = template.render(statements=statements, count=n)

        response = self.model.generate(prompt, max_tokens=200, stop=["###"])
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        focal_points = [
            re.sub(r"^[0-9]+[\).\s]*", "", line).strip() for line in lines
        ]
        logger.debug(f"[Reflection] Focal points: {focal_points}")
        return focal_points[:n]

    def _generate_insights_and_evidence(
        self, agent: Agent, nodes: List[MemoryNode], n: int = 5
    ) -> Dict[str, List[str]]:
        statements = "\n".join(
            f"{idx}. {node.event.description}" for idx, node in enumerate(nodes)
        )

        template = self.env.get_template("insight_and_evidence.txt")
        prompt = template.render(statements=statements, target=agent.name, count=n)

        response = self.model.generate(prompt, max_tokens=400, stop=["###"])
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

        insights: Dict[str, List[str]] = {}
        for line in lines:
            line = re.sub(r"^[0-9]+[\).\s]*", "", line)
            match = re.match(r"(.+?)\s*\(because of ([0-9,\s]+)\)", line)
            if match:
                thought = match.group(1).strip()
                indices = [
                    int(i.strip())
                    for i in match.group(2).split(",")
                    if i.strip().isdigit()
                ]
                evidence_ids = [
                    nodes[i].node_id for i in indices if 0 <= i < len(nodes)
                ]
            else:
                thought = line.strip()
                evidence_ids = []

            insights[thought] = evidence_ids

        logger.debug(f"[Reflection] Insights: {insights}")
        return insights

    def _thought_to_event(self, agent: Agent, thought: str) -> Event:
        triples = self.event_generator.generate_event(agent, thought)
        if triples:
            subject, predicate, obj = triples[0]
        else:
            subject, predicate, obj = agent.name, "is", "thinking"
        return Event(subject=subject, predicate=predicate, object=obj, description=thought)

    def _thought_significance(self, agent: Agent, thought: str) -> float:
        template = self.env.get_template("thought_significance.txt")
        context = {"agent": agent.get_state(), "thought": thought}
        prompt = template.render(context)

        try:
            response = self.model.generate(prompt, max_tokens=10, stop=["\n", ">>"])
            score = float(response.strip().lstrip(">").strip())
            score = max(min(score, 10.0), 1.0)
        except Exception:
            logger.warning("[Reflection] Failed to parse thought significance, defaulting to 1")
            score = 1.0
        return score

    def _reset_reflection_counter(self, agent: Agent) -> None:
        stm = agent.short_term_memory
        stm.importance_trigger_current = self.importance_threshold
        stm.importance_element_count = 0

