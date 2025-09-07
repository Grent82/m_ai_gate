from jinja2 import Environment, FileSystemLoader
from models.local_model import LocalModel
from core.agent import Agent
from core.logger import setup_logger

logger = setup_logger(__name__)

class InnerThoughtGenerator:
    def __init__(self, model: LocalModel, prompt_dir: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(prompt_dir))

    def generate_inner_thought(self, agent: Agent, whisper: str) -> str:
        logger.info("[InnerThought] Generating inner thought...")
        
        template = self.env.get_template("whisper_inner_thought.txt")

        context = {
            "agent": agent.get_state(),
            "whisper": whisper
        }

        prompt = template.render(context)

        #logger.debug(f"[InnerThought] Rendered prompt:\n{prompt}")

        response = self.model.generate(prompt, max_tokens=100, stop=["</thought>", "User:", "###"])
        logger.debug(f"[InnerThought] Raw LLM response: {response}")

        # Optionally clean the output
        return response.strip().replace("<thought>", "").replace("</thought>", "").strip()
