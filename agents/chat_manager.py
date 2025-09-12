from typing import List, Optional
import re
from jinja2 import Environment, FileSystemLoader
from models.local_model import LocalModel
from core.logger import setup_logger
from core.world import World
from core.agent import Agent

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

    def generate_conversation(self, agent: Agent, world: World, target: str, max_turns: int = 6) -> List[List[str]]:
        target_agent = self._find_agent_by_name(world, target)
        agent_state = agent.get_state()
        target_state = target_agent.get_state() if target_agent else {"name": target, "traits": "", "lifestyle": "", "status": "", "memories": []}

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
            "agent_thoughts_about_target": agent_th_about_target,
            "target_thoughts_about_agent": target_th_about_agent,
            "max_turns": max_turns,
        }
        prompt = self.env.get_template("generate_conversation.txt").render(context)
        raw = self.model.generate(prompt, max_tokens=300, stop=["</conversation>", "###"]).strip()
        convo: List[List[str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            speaker, utterance = line.split(":", 1)
            convo.append([speaker.strip(), utterance.strip()])
            logger.debug(f"[Chat] Line => {speaker.strip()}: {utterance.strip()}")
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
