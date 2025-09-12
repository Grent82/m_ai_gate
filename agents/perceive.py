from typing import List, Optional, Set, Tuple

from jinja2 import Environment, FileSystemLoader

from core.agent import Agent
from core.agent_action import Event
from core.tile import Tile
from core.world import World
from memory.long_term_memory import MemoryNode
from models.embeddings import get_embedding
from models.local_model import LocalModel
from core.logger import setup_logger

logger = setup_logger(__name__, log_level="DEBUG")


class Perception:
    def __init__(self, model: LocalModel, template_path: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(template_path))
        self.agent: Optional[Agent] = None
        self.world: Optional[World] = None

    def perceive(self, agent: Agent, world: World) -> List[MemoryNode]:
        logger.info("[Perception] Running perception...")

        self._initialize(agent, world)

        memory_nodes: List[MemoryNode] = []
        chat_node: Optional[MemoryNode] = None

        try:
            nearby_tiles = world.tile_manager.get_nearby_tiles_positions(
                agent.position, agent.vision_range
            )
            #logger.debug(f"[Perception] Nearby tiles: {nearby_tiles}")
            self._store_spatial_memory(nearby_tiles)

            nearest_events = self._gather_events_near_agent(nearby_tiles)

            chat_node = self._process_self_chat(nearest_events)

            if chat_node:
                memory_nodes.append(chat_node)

            memory_nodes.extend(
                self._store_significant_percepts(nearest_events, chat_node)
            )
        except Exception as e:
            logger.error(f"Error during perception: {str(e)}")

        return memory_nodes
    
    def _initialize(self, agent: Agent, world: World):
        self.agent = agent
        self.world = world

    def _store_spatial_memory(self, tiles: List[Tuple[int, int]]) -> None:
        #logger.debug("[Perception] Store spatial memory...")
        for tile_pos in set(tiles):
            tile_data = self._get_tile_data(tile_pos)
            if not tile_data:
                continue
            # Only store well-defined spatial info
            world_name = tile_data.world_name
            sector = tile_data.sector
            arena = tile_data.arena
            game_object = tile_data.game_object

            if not world_name or not sector or not arena:
                continue

            # Ensure sector/arena presence and add object if available
            self.agent.spatial_memory.update_memory(world_name, sector, [arena])
            if game_object:
                self.agent.spatial_memory.update_arena_objects(
                    world_name, sector, arena, [game_object]
                )
            
            #logger.debug("[Perception] Spatial update: world='%s', sector='%s', arena='%s', object='%s'",world_name,sector,arena,game_object or "",)

    def _gather_events_near_agent(self, tiles: List[Tuple[int, int]]) -> List[Event]:
        logger.debug("[Perception] Gather events...")
        current_arena = self.world.tile_manager.get_tile_path(
            self.agent.position, "arena"
        )
        seen_tuples = set()
        percept_events = []

        for tile_pos in tiles:
            if (
                self.world.tile_manager.get_tile_path(tile_pos, "arena")
                != current_arena
            ):
                continue

            tile_data = self._get_tile_data(tile_pos)
            if not tile_data:
                logger.warning(f"[Perception] No tile data at {tile_pos}")
                continue

            for event in getattr(tile_data, "events", []):
                logger.debug(f"[Perception] Found event at {tile_pos}: {event}")
                event = self._normalize_event(event)
                triple = (event.subject, event.predicate, event.object)
                if triple in seen_tuples:
                    continue
                seen_tuples.add(triple)
                distance = self.world.calculate_distance(self.agent.position, tile_pos)
                percept_events.append((distance, event))

                logger.debug(
                    f"[Perception] Added event: {event} at distance {distance}"
                )

        percept_events.sort(key=lambda x: x[0])
        logger.debug(
            f"[Perception] Sorted scored events (top {self.agent.attention_bandwidth}): {percept_events}"
        )
        selected_events = [event for _, event in percept_events[:self.agent.attention_bandwidth]]
        logger.debug(f"[Perception] Selected events (sorted): {selected_events}")
        return selected_events

    def _process_self_chat(
        self, events: List[Event]
    ) -> Optional[MemoryNode]:
        logger.debug("[Perception] Process self-chat...")
        latest_events = self.agent.long_term_memory.get_summarized_latest_events(self.agent.retention)

        for event in events:
            if event in latest_events:
                continue

            if event.subject == self.agent.name and event.predicate == "chat with":
                short_term = self.agent.short_term_memory
                action_event = short_term.action.event

                if not action_event or not action_event.description:
                    logger.debug("[Perception] No valid action event or description found.")
                    return None
                
                keywords = self._extract_keywords(event.subject, event.object)

                if action_event.description: 
                    logger.debug(f"[Perception] Current self-chat {action_event.description}")

                embedding = get_embedding(action_event.description)
                significance = self._calculate_significance(action_event.description, "chat")

                return self.agent.long_term_memory.add_chat(
                    action_event,
                    relevance=significance,
                    keywords=keywords,
                    filling=short_term.action.chat.chat_log,
                    embedding=embedding,
                )
        logger.debug("[Perception] No self-chat found")
        return None

    def _store_significant_percepts(
        self, events: List[Event], chat_node: Optional[MemoryNode]
    ) -> List[MemoryNode]:
        #logger.debug("[Perception] Store significant percepts...")
        latest_events = self.agent.long_term_memory.get_summarized_latest_events(self.agent.retention)
        chat_node_ids = [chat_node.node_id] if chat_node else []

        nodes: List[MemoryNode] = []
        for event in events:
            if event in latest_events:
                continue
            if not event.description or event.description.strip().lower() == "idle":
                continue
            
            logger.debug(f"[Perception] Storing percepts for event='{event}' and chat node is: '{chat_node_ids}'")
            node = self._store_percept("event", event, chat_node_ids)
            if node:
                nodes.append(node)
        return nodes

    def _store_percept(
        self,
        percept_type: str,
        event: Event,
        chat_node_ids: Optional[List[str]] = None,
    ) -> Optional[MemoryNode]:
        #logger.debug("[Perception] Store percept...")
        significance = self._calculate_significance(event.description, percept_type)

        keywords = self._extract_keywords(event.subject, event.object)
        embedding = get_embedding(event.description)

        self.agent.short_term_memory.importance_trigger_current -= significance
        self.agent.short_term_memory.importance_element_count += 1

        logger.debug(
            f"[Perception] Storing event: {event} with keywords='{keywords}' and chat_node_ids='{chat_node_ids}'"
        )
        return self.agent.long_term_memory.add_event(
            event=event,
            relevance=significance,
            keywords=keywords,
            embedding=embedding,
            filling=chat_node_ids,
        )

    def _calculate_significance(self, description: str, percept_type: str) -> float:
        #logger.debug("[Perception] Calculate significance...")
        context = {"agent": self.agent.get_state(), "description": description}
        template_file = (
            "event_significance.txt"
            if percept_type == "event"
            else "chat_significance.txt"
        )
        prompt = self._render_prompt(template_file, context)

        try:
            response = self.model.generate(prompt, max_tokens=10, stop=["\n", ">>"])
            score = float(response.strip().lstrip(">").strip())
            score = max(min(score, 10.0), 1.0)

            #logger.debug(f"[Perception] Prompt used for significance: {prompt}")
            #logger.debug(f"[Perception] Raw model response: '{response.strip()}' => Score: {score}")
        except Exception as e:
            logger.warning(f"[Perception] Significance defaulted: {e}")
            score = 1.0

        for trigger in self.agent.importance_triggers:
            if trigger.lower() in description.lower():
                score = min(score + 2.0, 10.0)
        return score

    def _extract_keywords(self, subject: str, obj: str) -> Set[str]:
        return {k for k in [self._normalize(subject), self._normalize(obj)] if k}

    def _normalize_event(self, event: Event) -> Event:
        return Event(
            subject=event.subject,
            predicate=event.predicate or "is",
            object=event.object or "idle",
            description=event.description or "idle"
        )

    def _normalize(self, value: Optional[str]) -> str:
        return value.split(":")[-1].strip().strip("()") if value else ""

    def _get_tile_data(self, tile_pos: Tuple[int, int]) -> Optional[Tile]:
        try:
            return self.world.tile_manager.get_tile(*tile_pos)
        except Exception as e:
            logger.error(f"Error retrieving tile data for {tile_pos}: {str(e)}")
            return None

    def _render_prompt(self, template_name: str, context: dict) -> str:
        return self.env.get_template(template_name).render(**context)
