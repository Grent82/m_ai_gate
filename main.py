import os
import random
from dotenv import load_dotenv

from agents.perceive import Perception
from agents.planner import ModularPlanner
from agents.reflection import Reflection
from agents.executor import Executor
from agents.retrieval import Retrieval
from agents.event_triple_generator import EventTripleGenerator
from agents.inner_thought_generator import InnerThoughtGenerator
from core.setup.agent_builder import AgentBuilder
from core.setup.world_builder import WorldBuilder
from models.local_model import LocalModel
from core.logger import setup_logger


def run_agent() -> None:
    """Bootstraps the world, agents, and runs the simulation tick loop."""

    # Load environment and set deterministic seeds
    load_dotenv()
    random.seed(42)

    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = setup_logger(__name__, log_level=log_level)

    # Simulation timing (minutes)
    tick_minutes = int(os.getenv("TICK_MINUTES", "5"))
    sim_minutes = int(os.getenv("SIM_MINUTES", "1440"))
    ticks = sim_minutes // tick_minutes

    model = LocalModel()
    perception = Perception(model)
    planner = ModularPlanner(model)
    executor = Executor(model)

    world = WorldBuilder().setup_medieval_village_world()
    world.current_time = world.current_time.replace(hour=15, minute=0)

    agents = AgentBuilder(InnerThoughtGenerator(model), EventTripleGenerator(model)).build_medieval_agents()
    for agent in agents:
        world.add_agent(agent)


    for tick in range(ticks):
        logger.info(f"=== Tick {tick + 1}/{ticks} @ {world.current_time.strftime('%H:%M')} ===")
        for agent in agents:
            agent.short_term_memory.current_time = world.current_time
            # Cleanup expired action events before perception/planning
            try:
                cleaned = executor.cleanup_expired(agent, world)
                if cleaned:
                    logger.debug(f"Cleaned up expired action for {agent.name}.")
            except Exception:
                pass

            perceived = perception.perceive(agent, world)
            logger.info(f"Perceived {len(perceived)} events for {agent.name}.")

            retrieval = Retrieval(agent.long_term_memory, agent.short_term_memory)
            retrieved = retrieval.retrieve_context(perceived)
            logger.info(f"Retrieved {len(retrieved)} relevant memory nodes for {agent.name}.")

            plan = planner.plan(agent, world, retrieved, perceived)
            logger.info(f"Plan for {agent.name}: {plan}")

            reflection = Reflection(model, retrieval)
            reflections = reflection.reflect(agent)
            logger.info(f"Generated {len(reflections)} reflective thoughts for {agent.name}.")

            status = executor.execute(agent, world, max_steps=2)
            logger.info(f"Execution status for {agent.name}: {status}")

        # Advance global time after all agents act
        world.advance_time(tick_minutes)

if __name__ == "__main__":
    run_agent()
