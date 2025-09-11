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


def run_agent():

    logger = setup_logger(__name__, log_level="DEBUG")

    model = LocalModel()
    perception = Perception(model)
    planner = ModularPlanner(model)
    executor = Executor(model)

    world = WorldBuilder().setup_medieval_village_world()
    agents = AgentBuilder(InnerThoughtGenerator(model), EventTripleGenerator(model)).build_medieval_agents()
    for agent in agents:
        world.add_agent(agent)


    # Simulate time progression across ticks
    TICKS = 3600 // 5
    TICK_MINUTES = 5
    for tick in range(TICKS):
        logger.info(f"=== Tick {tick + 1}/{TICKS} @ {world.current_time.strftime('%H:%M')} ===")
        for agent in agents:
            # Sync time
            agent.short_term_memory.current_time = world.current_time

            # Perceive current world
            perceived = perception.perceive(agent, world)
            logger.info(f"Perceived {len(perceived)} events for {agent.name}.")

            # Retrieve and plan
            retrieval = Retrieval(agent.long_term_memory, agent.short_term_memory)
            retrieved = retrieval.retrieve_context(perceived)
            logger.info(f"Retrieved {len(retrieved)} relevant memory nodes for {agent.name}.")

            # Cleanup any expired action before planning
            executor.cleanup_expired(agent, world)

            plan = planner.plan(agent, world, retrieved, perceived)
            logger.info(f"Plan for {agent.name}: {plan}")

            # Execute a few steps of the plan
            status = executor.execute(agent, world, max_steps=3)
            logger.info(f"Execution status for {agent.name}: {status}")

            # Perceive again to store the outcome
            post_exec_percepts = perception.perceive(agent, world)
            logger.info(f"Post-exec perceived {len(post_exec_percepts)} events for {agent.name}.")

            # Reflect
            reflact = Reflection(model, retrieval)
            reflections = reflact.reflect(agent)
            logger.info(f"Generated {len(reflections)} reflective thoughts for {agent.name}.")

        # Advance global time after all agents act
        world.advance_time(TICK_MINUTES)


    # === Modular Planning ===
    #planner.plan(agent, world)

    #tile_actions = planner.get_tile_actions(agent)
    #print(tile_actions)

if __name__ == "__main__":
    run_agent()
