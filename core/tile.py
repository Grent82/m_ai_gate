from typing import List

from core.agent_action import Event
from core.logger import setup_logger

logger = setup_logger(__name__)

class Tile:
    def __init__(self, world_name, sector, arena, game_object, spawning_location, collision=False):
        self.world_name = world_name
        self.sector = sector
        self.arena = arena
        self.game_object = game_object
        self.spawning_location = spawning_location
        self.collision = collision
        self.events: List[Event] = []

    def add_event(self, event: Event):
        self.events.append(event)

    def remove_event(self, event: Event):
        if event in self.events:
            self.events.remove(event)

    def has_event(self, event: Event) -> bool:
        return event in self.events

    def is_collidable(self) -> bool:
        return self.collision

    def __repr__(self):
        events_repr = ", ".join(repr(event) for event in self.events)
        return (
            f"Tile(world={self.world_name}, sector={self.sector}, arena={self.arena}, "
            f"game_object={self.game_object}, spawning_location={self.spawning_location}, "
            f"collision={self.collision}, events=[{events_repr}])"
        )
