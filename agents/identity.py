from jinja2 import Environment, FileSystemLoader
from models.local_model import LocalModel
from core.agent_action import Event
from models.embeddings import get_embedding
from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")


class IdentityReviser:
    def __init__(self, model: LocalModel, template_path: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(template_path))

    def revise(self, agent) -> None:
        context = {
            "agent": agent.get_state(),
        }
        prompt = self.env.get_template("revise_identity_status.txt").render(context)
        raw = self.model.generate(prompt, max_tokens=200, stop=["\n", "</status>", "###"]).strip()
        if raw:
            agent.status = raw
            # Store as a thought
            thought = f"Status update for {agent.name}: {raw}"
            embedding = get_embedding(thought)
            event = Event(agent.name, "status", agent.name, thought)
            agent.long_term_memory.add_thought(event, relevance=5, keywords={"status"}, filling=None, embedding=embedding)
            logger.debug("[Identity] Revised status: %s", raw)

