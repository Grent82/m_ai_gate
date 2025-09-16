from typing import List, Optional
from enum import Enum

from core.agent_action import Event
from core.logger import setup_logger

logger = setup_logger(__name__)


class TerrainType(Enum):
    """
    Terrain types with associated navigation cost.

    Costs are relative multipliers; 1.0 is normal ground.
    """
    FLOOR = ("floor", 1.0)
    WALL = ("wall", 1.0)
    STONE = ("stone", 1.0)
    GRASS = ("grass", 1.0)
    SOIL = ("soil", 1.0)
    GRAVEL = ("gravel", 0.5)
    FOREST = ("forest", 1.2)
    WATER = ("water", 2.0)

    def __init__(self, label: str, cost: float):
        self.label = label
        self.cost = cost

class Tile:
    def __init__(self, world_name, sector, arena, game_object, spawning_location, collision=False):
        self.world_name = world_name
        self.sector = sector
        self.arena = arena
        self.game_object = game_object
        self.spawning_location = spawning_location
        self.collision = collision
        self.terrain: Optional[TerrainType] = None
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

    def nav_cost(self) -> float:
        """Navigation cost derived from terrain (defaults to 1.0)."""
        try:
            return self.terrain.cost if self.terrain is not None else 1.0
        except Exception:
            return 1.0

    def __repr__(self):
        events_repr = ", ".join(repr(event) for event in self.events)
        return (
            f"Tile(world={self.world_name}, sector={self.sector}, arena={self.arena}, "
            f"terrain={getattr(self.terrain, 'label', None)}, game_object={self.game_object}, "
            f"spawning_location={self.spawning_location}, collision={self.collision}, "
            f"events=[{events_repr}])"
        )
