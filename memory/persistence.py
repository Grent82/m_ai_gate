import os

from core.agent import Agent
from core.logger import setup_logger
from .long_term_memory import LongTermMemory
from .short_term_memory import ShortTermMemory

logger = setup_logger(__name__, log_level="DEBUG")


def _agent_dir(base_dir: str, agent_name: str) -> str:
    safe_name = "_".join((agent_name or "agent").split())
    return os.path.join(base_dir, safe_name)


def save_agent_state(agent: Agent, base_dir: str = "data") -> None:
    d = _agent_dir(base_dir, agent.name)
    os.makedirs(d, exist_ok=True)

    # Long-term memory
    ltm_dir = os.path.join(d, "ltm")
    os.makedirs(ltm_dir, exist_ok=True)
    agent.long_term_memory.save(ltm_dir)

    # Short-term memory
    stm_path = os.path.join(d, "stm.json")
    agent.short_term_memory.save(stm_path)


def load_agent_state(agent: Agent, base_dir: str = "data") -> None:
    d = _agent_dir(base_dir, agent.name)
    ltm_dir = os.path.join(d, "ltm")
    stm_path = os.path.join(d, "stm.json")

    # Load LTM if present
    if os.path.exists(os.path.join(ltm_dir, "nodes.json")):
        try:
            loaded_ltm = LongTermMemory.load(ltm_dir)
            agent.long_term_memory = loaded_ltm
        except Exception:
            logger.error(f"Failed to load LTM for agent {agent.name} from {ltm_dir}")
            pass

    # Load STM if present
    if os.path.exists(stm_path):
        try:
            agent.short_term_memory = ShortTermMemory.load(stm_path)
        except Exception:
            logger.error(f"Failed to load STM for agent {agent.name} from {stm_path}")
            pass

