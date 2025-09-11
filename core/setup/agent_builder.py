from datetime import datetime, timedelta
from core.agent import Agent
from core.agent_action import Event
from agents.event_triple_generator import EventTripleGenerator
from agents.inner_thought_generator import InnerThoughtGenerator
from models.embeddings import get_embedding

class AgentBuilder:
    def __init__(self, inner_thought_generator: InnerThoughtGenerator, event_triple_generator: EventTripleGenerator):
        self.inner_thought_generator = inner_thought_generator
        self.event_triple_generator = event_triple_generator

    def build_medieval_agents(self) -> list:
        agents = []

        def create_agent(name, age, traits, lifestyle, position, memories, whisper, background, status, sex: str):
            agent = Agent(
                name=name,
                age=age,
                traits=traits,
                lifestyle=lifestyle,
                position=position,
                background=background,
                status=status,
                sex=sex,
            )
            agent.short_term_memory.current_time = datetime.now()

            for content, relevance in memories:
                agent.add_long_term_memory(content, relevance)

            thought = self.inner_thought_generator.generate_inner_thought(agent, whisper)
            triples = self._parse_event_triple(agent, thought)
            for s, p, o in triples:
                embedding = get_embedding(thought)
                agent.long_term_memory.add_thought(
                    event=self._make_event(s, p, o, thought),
                    relevance=6.5,
                    keywords={s, p, o},
                    filling=[[s, p, o]],
                    embedding=embedding
                )
            return agent

        agents.append(create_agent(
            name="Thomas the Farmer",
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
                "You have a crush on Ayla."
            ),
            background=(
                "A farmer in a tiny forest hamlet; inherited his father's barley and cabbage field beside his house."
            ),
            status="content, slightly tired from fieldwork",
            sex="male"
        ))

        agents.append(create_agent(
            name="Ayla the Huntress",
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
                "A solitary huntress living at the forest's edge; skilled with bow and traps; fiercely self-sufficient."
            ),
            status="alert, reserved",
            sex="female"
        ))

        agents.append(create_agent(
            name="Garrick the Innkeeper",
            age=52,
            traits="jovial, talkative, wise",
            lifestyle="Wakes late, cleans the tavern, chats with guests, and enjoys telling stories in the evening.",
            position=(6, 11),
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
                "You have a crush on Ayla."
            ),
            background=(
                "Innkeeper of The Drunken Boar Tavern in a tiny forest hamlet; keeper of stories and village gossip."
            ),
            status="cheerful, attentive to guests",
            sex="male"
        ))

        return agents

    def _parse_event_triple(self, agent: Agent, description: str):
        return self.event_triple_generator.generate_event(agent, description)

    def _make_event(self, s, p, o, description):
        return Event(subject=s, predicate=p, object=o, description=description)
