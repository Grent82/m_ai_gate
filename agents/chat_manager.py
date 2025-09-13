from typing import List, Optional, Iterable
import re
from jinja2 import Environment, FileSystemLoader
from models.local_model import LocalModel
from core.logger import setup_logger
from core.world import World
from core.agent import Agent
from core.agent_action import Event
from models.embeddings import get_embedding

logger = setup_logger(__name__, log_level="DEBUG")


class ChatManager:
    def __init__(self, model: LocalModel, template_path: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(template_path))

    def _find_agent_by_name(self, world: World, name: str) -> Optional[Agent]:
        try:
            return next((a for a in world.agents if a.name == name), None)
        except Exception:
            return None

    def _latest_chat_summary(self, agent: Agent, other_name: str) -> Optional[str]:
        try:
            for node in agent.long_term_memory.node_sequences.get("chat", []):
                ev = getattr(node, "event", None)
                if not ev:
                    continue
                if (ev.subject == agent.name and ev.object == other_name) or (
                    ev.subject == other_name and ev.object == agent.name
                ):
                    return ev.description
        except Exception:
            pass
        return None

    def _recent_thoughts_about(self, agent: Agent, other_name: str, limit: int = 2) -> Optional[str]:
        thoughts: List[str] = []
        try:
            for node in agent.long_term_memory.node_sequences.get("thought", []):
                ev = getattr(node, "event", None)
                if not ev or not ev.description:
                    continue
                if other_name.lower() in ev.description.lower():
                    thoughts.append(ev.description)
                if len(thoughts) >= limit:
                    break
        except Exception:
            pass
        return "; ".join(thoughts[:limit]) if thoughts else None

    def _gather_relationship_statements(
        self, agent: Agent, target_agent: Optional[Agent], target_name: str, max_items: int = 12
    ) -> List[str]:
        """Collect short statements relevant to the pair's relationship.

        Pulls from both agents' thoughts, events, and past chat summaries.
        """
        names = {agent.name.lower(), target_name.lower()}

        def nodes_iter(a: Agent) -> Iterable:
            for t in ("thought", "event", "chat"):
                for n in a.long_term_memory.node_sequences.get(t, []):
                    yield n

        lines: List[str] = []

        def add_line(s: str) -> None:
            s = (s or "").strip()
            if not s:
                return
            if s.lower() in {x.lower() for x in lines}:
                return
            lines.append(s)

        for a in filter(None, [agent, target_agent]):
            for node in nodes_iter(a):
                ev = getattr(node, "event", None)
                if not ev:
                    continue
                desc = (ev.description or "").strip()
                subj = (ev.subject or "").strip()
                obj = (ev.object or "").strip()
                pair_mentioned = (
                    any(n in (desc + " " + subj + " " + obj).lower() for n in names)
                )
                if not pair_mentioned:
                    continue
                if node.node_type == "chat":
                    if desc:
                        add_line(desc)
                    try:
                        fill = getattr(node, "filling", None) or []
                        if fill:
                            tail = fill[-2:] if len(fill) >= 2 else fill
                            add_line(" | ".join(f"{s}: {u}" for s, u in tail))
                    except Exception:
                        pass
                else:
                    if desc:
                        add_line(desc)
                    elif subj or obj:
                        add_line(f"{subj} {ev.predicate or 'is'} {obj}")

                if len(lines) >= max_items:
                    break
            if len(lines) >= max_items:
                break
        return lines[:max_items]

    def summarize_relationship(self, agent: Agent, world: World, target: str) -> Optional[str]:
        target_agent = self._find_agent_by_name(world, target)
        statements = self._gather_relationship_statements(agent, target_agent, target)
        if not statements:
            return None

        context = {
            "agent": agent.get_state(),
            "target_agent": target_agent.get_state() if target_agent else {
                "name": target,
                "age": "",
                "traits": "",
                "lifestyle": "",
                "status": "",
                "background": "",
                "memories": [],
            },
            "statements": "\n".join(statements),
        }
        try:
            logger.debug(f"[Relation] Found {len(statements)} statements for {agent.name} ↔ {target}")
        except Exception:
            pass
        prompt = self.env.get_template("summarize_relationship.txt").render(context)
        summary = self.model.generate(prompt, max_tokens=80, stop=["\n", "</summary>", "###"]).strip()
        if not summary:
            return None

        # Store as a thought for future retrieval on both agents (if present)
        try:
            ev_a = Event(agent.name, "relationship with", target_agent.name if target_agent else target, summary)
            emb_a = get_embedding(summary)
            agent.long_term_memory.add_thought(ev_a, relevance=5.0, keywords={agent.name, target.lower(), "relationship"}, filling=None, embedding=emb_a)
            logger.debug(f"[Relation] Stored relationship summary for {agent.name} → {target}")
        except Exception:
            pass
        try:
            if target_agent:
                ev_b = Event(target_agent.name, "relationship with", agent.name, summary)
                emb_b = get_embedding(summary)
                target_agent.long_term_memory.add_thought(ev_b, relevance=5.0, keywords={target_agent.name, agent.name.lower(), "relationship"}, filling=None, embedding=emb_b)
                logger.debug(f"[Relation] Stored relationship summary for {target_agent.name} → {agent.name}")
        except Exception:
            pass
        try:
            logger.debug(f"[Relation] Summary => {summary}")
        except Exception:
            pass
        return summary

    def generate_conversation(self, agent: Agent, world: World, target: str, max_turns: int = 6) -> List[List[str]]:
        target_agent = self._find_agent_by_name(world, target)
        agent_state = agent.get_state()
        target_state = (
            target_agent.get_state()
            if target_agent
            else {
                "name": target,
                "age": "",
                "traits": "",
                "lifestyle": "",
                "status": "",
                "background": "",
                "memories": [],
            }
        )

        sector_path = ""
        arena_path = ""
        try:
            sector_path = world.tile_manager.get_tile_path(agent.position, "sector")
            arena_path = world.tile_manager.get_tile_path(agent.position, "arena")
        except Exception:
            pass
        def last_part(path: str) -> str:
            try:
                return path.split(":")[-1]
            except Exception:
                return path or "Unknown"

        prior = self._latest_chat_summary(agent, target)
        relationship = None
        try:
            relationship = self.summarize_relationship(agent, world, target)
        except Exception:
            relationship = None
        try:
            logger.debug(f"[Chat] Relationship context present: {bool(relationship)} for {agent.name} ↔ {target}")
        except Exception:
            pass

        # Find latest chat node between the two (for time + excerpt)
        last_chat_time: Optional[str] = None
        last_chat_excerpt: Optional[str] = None
        try:
            for node in agent.long_term_memory.node_sequences.get("chat", []):
                ev = getattr(node, "event", None)
                if not ev:
                    continue
                if (ev.subject == agent.name and ev.object == (target_agent.name if target_agent else target)) or (
                    ev.subject == (target_agent.name if target_agent else target) and ev.object == agent.name
                ):
                    last_chat_time = getattr(node, "created", None)
                    if last_chat_time:
                        last_chat_time = last_chat_time.strftime("%H:%M")
                    # node.filling may carry chat lines like [[speaker, utterance], ...]
                    lines = []
                    try:
                        fill = getattr(node, "filling", None) or []
                        # Take the last 2 lines for brevity
                        for s, u in (fill[-2:] if len(fill) >= 2 else fill):
                            lines.append(f"{s}: {u}")
                    except Exception:
                        pass
                    if lines:
                        last_chat_excerpt = "\n".join(lines)
                    break
        except Exception:
            pass
        agent_th_about_target = self._recent_thoughts_about(agent, target)
        target_th_about_agent = self._recent_thoughts_about(target_agent, agent.name) if target_agent else None

        context = {
            "agent": agent_state,
            "target_agent": target_state,
            "world": world.get_state(),
            "location": {"sector": last_part(sector_path), "arena": last_part(arena_path)},
            "agent_action": getattr(agent.short_term_memory.action, "description", None),
            "target_action": getattr(target_agent.short_term_memory.action, "description", None) if target_agent else None,
            "prior_summary": prior,
            "relationship": relationship,
            "last_chat_time": last_chat_time,
            "last_chat_excerpt": last_chat_excerpt,
            "agent_thoughts_about_target": agent_th_about_target,
            "target_thoughts_about_agent": target_th_about_agent,
            "max_turns": max_turns,
        }
        prompt = self.env.get_template("generate_conversation.txt").render(context)
        raw = self.model.generate(prompt, max_tokens=300, stop=["</conversation>", "###"]).strip()

        # Sanitize: strip special tokens and only accept exact speakers
        def _strip_tags(s: str) -> str:
            try:
                return re.sub(r"<\|[^>]+\|>", "", s)
            except Exception:
                return s

        allowed = {agent.name.lower(), (target_agent.name.lower() if target_agent else target.lower())}
        convo: List[List[str]] = []
        for line in raw.splitlines():
            line = _strip_tags(line).strip()
            if not line or ":" not in line:
                continue
            speaker, utterance = line.split(":", 1)
            sp = speaker.strip().strip('"“”').lower()
            if sp not in allowed:
                continue
            utt = _strip_tags(utterance).strip().strip('"“”')
            if not utt or utt.lower() == "<placeholder>":
                continue
            convo.append([speaker.strip(), utt])
            logger.debug(f"[Chat] Line => {speaker.strip()}: {utt}")
        logger.debug(f"[Chat] Generated {len(convo)} convo lines.")
        return convo[: max_turns]

    def summarize_conversation(self, agent, convo: List[List[str]]) -> str:
        convo_text = "\n".join(f"{s}: {u}" for s, u in convo)
        context = {
            "agent": agent.get_state(),
            "dialogue": convo_text,
        }
        prompt = self.env.get_template("summarize_conversation.txt").render(context)
        summary = self.model.generate(prompt, max_tokens=120, stop=["\n", "</summary>", "###"]).strip()
        try:
            summary = re.sub(r"^(?:answer|summary|response)\s*:\s*", "", summary, flags=re.IGNORECASE).strip()
            logger.debug(f"[Chat] Summary => {summary}")
        except Exception:
            pass
        return summary
