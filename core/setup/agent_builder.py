from datetime import datetime, timedelta
from core.agent import Agent
from core.agent_action import Event
from agents.event_triple_generator import EventTripleGenerator
from agents.inner_thought_generator import InnerThoughtGenerator
from models.embedding_cache import get_embedding_cached
from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")

class AgentBuilder:
    def __init__(self, inner_thought_generator: InnerThoughtGenerator, event_triple_generator: EventTripleGenerator):
        self.inner_thought_generator = inner_thought_generator
        self.event_triple_generator = event_triple_generator

    def build_medieval_agents(self) -> list:
        agents = []

        def create_agent(
            first_name: str,
            last_name: str,
            sex: str,
            age,
            traits,
            lifestyle,
            position,
            memories,
            whisper,
            background,
            status,
            learned: str,
            living_area: str,
        ):
            agent = Agent(
                first_name=first_name,
                last_name=last_name,
                sex=sex,
                age=age,
                traits=traits,
                lifestyle=lifestyle,
                position=position,
                background=background,
                status=status,
                learned=learned,
                living_area=living_area,
            )
            agent.short_term_memory.current_time = datetime.now()

            for content, relevance in memories:
                agent.add_long_term_memory(content, relevance)

            thought = self.inner_thought_generator.generate_inner_thought(agent, whisper)
            triples = self._parse_event_triple(agent, thought)
            for s, p, o in triples:
                embedding = get_embedding_cached(thought)
                agent.long_term_memory.add_thought(
                    event=self._make_event(s, p, o, thought),
                    relevance=6.5,
                    keywords={s, p, o},
                    filling=[[s, p, o]],
                    embedding=embedding
                )
            return agent

        agents.append(create_agent(
            first_name="Thomas",
            last_name="Miller",
            sex="male",
            age=45,
            traits="hardworking, humble, kind",
            lifestyle="Wakes up early, tends to the fields, and eats dinner by the fireplace at sunset.",
            position=(3, 3),
            memories=[
                ("Owns a field next to his house where he grows barley and cabbage.", 8),
                ("Sells surplus crops to the tavern in the evening.", 5),
                ("Is friendly with the innkeeper and hunter.", 6)
            ],
            whisper=(
                "You deeply value the land inherited from your father; "
                "You feel a duty to keep the village fed; "
                "You trust Garrick the Innkeeper with village matters; "
                "You admire Ayla's independence even though she rarely speaks; "
                "You have a crush on Ayla; "
                "You suspect Garrick also seeks Ayla's favor; "
                "You feel a quiet rivalry with him but keep it respectful."
            ),
            background=(
                "A farmer in a tiny forest hamlet; inherited his father's barley and cabbage field beside his house; lives alone without family."
            ),
            status="content, slightly tired from fieldwork",
            learned=(
                "Thomas Miller is a diligent farmer skilled in plowing, sowing, and harvesting. "
            ),
            living_area="MedievalVillage:Farmer's House:Private Room",
        ))

        agents.append(create_agent(
            first_name="Ayla",
            last_name="Hart",
            sex="female",
            age=32,
            traits="silent, observant, resilient",
            lifestyle="Roams the forest at dawn, repairs gear in her cabin, and drinks occasionally at the tavern.",
            position=(10, 3),
            memories=[
                ("Skilled in trapping and archery, often hunts wild game in the forest.", 9),
                ("Has a secret stash hidden behind her cabin.", 4),
                ("Sometimes trades pelts and meat with the innkeeper.", 5)
            ],
            whisper=(
                "This is very important — you prefer solitude over company; "
                "You see Garrick as a source of useful gossip; "
                "You respect Thomas for his honesty; "
                "You hide your emotions and secrets in the forest."
            ),
            background=(
                "A solitary huntress living at the forest's edge; skilled with bow and traps; fiercely self-sufficient; lives alone without family."
            ),
            status="alert, reserved",
            learned=(
                "Ayla Hart is a capable huntress proficient in tracking, trapping, and bowcraft. "
            ),
            living_area="MedievalVillage:Hunter's Cabin:Private Room",
        ))

        agents.append(create_agent(
            first_name="Garrick",
            last_name="Brewer",
            sex="male",
            age=52,
            traits="jovial, talkative, wise",
            lifestyle="Wakes late, cleans the tavern, chats with guests, and enjoys telling stories in the evening.",
            position=(6, 10),
            memories=[
                ("Owns The Drunken Boar Tavern in the village center.", 7),
                ("Knows many rumors and secrets of the village.", 6),
                ("Respects the farmer and finds the hunter mysterious.", 5)
            ],
            whisper=(
                "You know everyone’s stories and enjoy collecting rumors; "
                "You often hear confessions late at night over ale; "
                "You like teasing Ayla, even if she rarely responds; "
                "You admire Thomas’s work ethic and often give him a free drink; "
                "You have a crush on Ayla; "
                "You sense Thomas also courts Ayla’s attention; "
                "You relish a subtle, friendly competition for her regard."
            ),
            background=(
                "Innkeeper of The Drunken Boar Tavern in a tiny forest hamlet; keeper of stories and village gossip; lives alone without family."
            ),
            status="cheerful, attentive to guests",
            learned=(
                "Garrick Brewer is an innkeeper skilled in serving ale, keeping accounts, and hosting guests. "
            ),
            living_area="MedievalVillage:The Drunken Boar Tavern:Private Room",
        ))

        return agents

    def _parse_event_triple(self, agent: Agent, description: str):
        logger.debug(f"[AgentBuilder] Parsing event triple for description '{description}'...")
        return self.event_triple_generator.generate_event(agent, description)

    def _make_event(self, s, p, o, description):
        return Event(subject=s, predicate=p, object=o, description=description)
