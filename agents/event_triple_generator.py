from typing import List, Tuple

from jinja2 import Environment, FileSystemLoader
from core.agent_action import Event
from models.local_model import LocalModel
from core.agent import Agent
from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")

class EventTripleGenerator:
    def __init__(self, model: LocalModel, prompt_dir: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(prompt_dir))

    def generate_event(self, agent: Agent, action_description: str) -> List[Tuple[str, str, str]]:
        template = self.env.get_template("generate_event_triple.txt")
        context = {
            "agent": agent.get_state(),
            "description": action_description
        }

        prompt = template.render(context)
        # logger.debug(f"[EventTriple] Prompt:\n{prompt}")

        response = self.model.generate(prompt, max_tokens=200, stop=["###", "</event>"])
        logger.debug(f"[EventTriple] Response: {response}")

        lines = response.strip().splitlines()
        triples = []
        for line in lines:
            if line.startswith("(") and line.endswith(")"):
                parts = line.strip("() ").split(",")
                if len(parts) == 3:
                    s, p, o = [part.strip().strip('"') for part in parts]
                    triples.append((s, p, o))
                else:
                    logger.warning(f"[EventTriple] Malformed triple: {line}")
        return triples
