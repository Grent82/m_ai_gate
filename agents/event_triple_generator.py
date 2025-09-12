from typing import List, Tuple
import re

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

        triples: List[Tuple[str, str, str]] = []

        # Extract any parenthesized tuple-like content anywhere in the text
        for m in re.finditer(r"\(([^)]+)\)", response):
            inner = m.group(1).strip()
            if inner.lower().startswith("note:"):
                continue
            parts = [p.strip().strip('"') for p in inner.split(",")]
            if len(parts) >= 3:
                s = parts[0]
                p = parts[1]
                o = ",".join(parts[2:]).strip()
                if s and p and o:
                    triples.append((s, p, o))
                continue
            if len(parts) == 2:
                s, rest = parts
                tokens = rest.split()
                if len(tokens) >= 2:
                    p = tokens[0]
                    o = " ".join(tokens[1:])
                else:
                    p = "is"
                    o = rest
                if s and p and o:
                    triples.append((s, p, o))

        # Fallback: key-value block format
        if not triples:
            try:
                subj = re.search(r"^\s*subject\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
                pred = re.search(r"^\s*predicate\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
                obj = re.search(r"^\s*object\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
                if subj and pred and obj:
                    s = subj.group(1).strip().strip('"')
                    p = pred.group(1).strip().strip('"')
                    o = obj.group(1).strip().strip('"')
                    triples.append((s, p, o))
            except Exception:
                pass
        return triples
