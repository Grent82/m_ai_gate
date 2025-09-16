from jinja2 import Environment, FileSystemLoader
from models.local_model import LocalModel
from core.agent_action import Event
from models.embeddings import get_embedding
from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")


class IdentityReviser:
    """Revise an agent's status using a local LLM guided by a Jinja2 template.

    This class renders a prompt from 'revise_identity_status.txt' (in ``template_path``),
    invokes :class:`models.local_model.LocalModel` to generate a status update, and, if
    produced, updates the agent and records the change in long‑term memory as an
    :class:`core.agent_action.Event` with an embedding from
    :func:`models.embeddings.get_embedding`.

    Parameters
    ----------
    model : LocalModel
        Local LLM interface used for text generation.
    template_path : str, optional
        Directory containing prompt templates, by default "prompts".

    Notes
    -----
    revise(agent) mutates the given agent (sets ``agent.status``) and appends a
    thought to ``agent.long_term_memory``; it returns None.
    """
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

