from typing import List
import re
from jinja2 import Environment, FileSystemLoader
from models.local_model import LocalModel
from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")


class ChatManager:
    def __init__(self, model: LocalModel, template_path: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(template_path))

    def generate_conversation(self, agent, target: str, max_turns: int = 6) -> List[List[str]]:
        context = {
            "agent": agent.get_state(),
            "target": target,
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
