from agents.perceive import Perception
from agents.planner import ModularPlanner
from agents.reflection import Reflection
from agents.retrieval import Retrieval
from agents.event_triple_generator import EventTripleGenerator
from agents.inner_thought_generator import InnerThoughtGenerator
from core.setup.agent_builder import AgentBuilder
from core.setup.world_builder import WorldBuilder
from models.local_model import LocalModel
from core.logger import setup_logger


def run_agent():

    logger = setup_logger(__name__, log_level="DEBUG")

    model = LocalModel()
    perception = Perception(model)
    planner = ModularPlanner(model)
    reflact = Reflection(model, Retrieval)

    world = WorldBuilder().setup_medieval_village_world()
    agents = AgentBuilder(InnerThoughtGenerator(model), EventTripleGenerator(model)).build_medieval_agents()
    for agent in agents:
        world.add_agent(agent)

        nearest = world.get_nearest_npc(agent)
        if nearest is None:
            logger.debug(f"No nearby agents for {agent.name} to interact with.")
        else:
            logger.debug(f"Nearest agent to {agent.name} is {nearest.name}.")

        retrieval = Retrieval(agent.long_term_memory, agent.short_term_memory)
        perceived = perception.perceive(agent, world)
        logger.info(f"Perceived {len(perceived)} events for {agent.name}.")
        retrieved = retrieval.retrieve_context(perceived)
        logger.info(f"Retrieved {len(retrieved)} relevant memory nodes for {agent.name}.")
        plan = planner.plan(agent, world, retrieved)
        logger.info(f"Plan for {agent.name}: {plan}")
        reflections = reflact.reflect(agent)
        logger.info(f"Generated {len(reflections)} reflective thoughts for {agent.name}.")


    # === Modular Planning ===
    #planner.plan(agent, world)

    #tile_actions = planner.get_tile_actions(agent)
    #print(tile_actions)

if __name__ == "__main__":
    run_agent()
