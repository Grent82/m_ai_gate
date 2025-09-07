from agents.perceive import Perception
from agents.planner import ModularPlanner
from agents.retrieval import Retrieval
from agents.event_triple_generator import EventTripleGenerator
from agents.inner_thought_generator import InnerThoughtGenerator
from core.setup.agent_builder import AgentBuilder
from core.setup.world_builder import WorldBuilder
from models.local_model import LocalModel


def run_agent():

    # model = LocalModel("/path/to/model.gguf")
    model = LocalModel()
    perception = Perception(model)
    
    planner = ModularPlanner(model)
    
    world = WorldBuilder().setup_medieval_village_world()
    agents = AgentBuilder(InnerThoughtGenerator(model), EventTripleGenerator(model)).build_medieval_agents()
    for agent in agents:
        world.add_agent(agent)
        retrieval = Retrieval(agent.long_term_memory, agent.short_term_memory)
        perceived = perception.perceive(agent, world)
        retrieved = retrieval.retrieve_context(perceived)
        planner.plan(agent, world, retrieved)

    # === Modular Planning ===
    #planner.plan(agent, world)

    #tile_actions = planner.get_tile_actions(agent)
    #print(tile_actions)

if __name__ == "__main__":
    run_agent()
