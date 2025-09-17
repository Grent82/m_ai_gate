from typing import List, Tuple
import re

from jinja2 import Environment, FileSystemLoader
from models.local_model import LocalModel
from core.agent import Agent
from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")


def _normalize_field(text: str) -> str:
    t = " ".join(text.strip().split())
    return t.rstrip(".;:,-")


class EventTripleGenerator:
    def __init__(self, model: LocalModel, prompt_dir: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(prompt_dir))

    def generate_event(self, agent: Agent, action_description: str) -> List[Tuple[str, str, str]]:
        template = self.env.get_template("generate_event_triple.txt")
        context = {
            "agent": agent.get_state(),
            "description": action_description,
        }

        prompt = template.render(context)
        logger.debug(f"[EventTriple] Prompt:\n{prompt}")

        response = self.model.generate(
            prompt,
            max_tokens=200,
            stop=["User:", "###"],
            temperature=0.3,
        )
        logger.debug(f"[EventTriple] Response: {response}")

        triples: List[Tuple[str, str, str]] = []
        
        for line in response.strip().splitlines():
            for m in re.finditer(r"\(([^)]+)\)", line):
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
                        logger.debug(f"[EventTriple] Extracted triple: {(s, p, o)}")
                elif len(parts) == 2:
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
                        logger.debug(f"[EventTriple] Extracted triple with 2 parts: {(s, p, o)}")

        # Try CSV extraction for each line
        if not triples:
            logger.debug("[EventTriple] Trying CSV extraction...")
            for line in response.strip().splitlines():
                m_csv = re.match(r"^\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*(.+?)\s*$", line)
                if m_csv:
                    s, p, o = m_csv.groups()
                    triples.append((s, p, o))
                    logger.debug(f"[EventTriple] Extracted CSV triple: {(s, p, o)}")

        # Try labeled field extraction (less likely for multiple triples)
        if not triples:
            logger.debug("[EventTriple] Trying labeled field extraction...")
            subj = re.search(r"^\s*subject\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
            pred = re.search(r"^\s*predicate\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
            obj = re.search(r"^\s*object\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
            if subj and pred and obj:
                s = subj.group(1).strip().strip('"')
                p = pred.group(1).strip().strip('"')
                o = obj.group(1).strip().strip('"')
                triples.append((s, p, o))
                logger.debug(f"[EventTriple] Extracted labeled triple: {(s, p, o)}")

        # Clean and normalize triples
        cleaned: List[Tuple[str, str, str]] = []
        for s, p, o in triples:
            s_n = _normalize_field(s)
            p_n = _normalize_field(p)
            o_n = _normalize_field(o)
            if s_n and p_n and o_n:
                cleaned.append((s_n, p_n, o_n))
                logger.debug(f"[EventTriple] Cleaned triple: {(s_n, p_n, o_n)}")

        return cleaned