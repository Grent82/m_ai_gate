from datetime import datetime

from core.tile_manager import TileManager
from core.logger import setup_logger

logger = setup_logger(__name__)

class World:
    def __init__(self, name, description, width, height):
        self.name = name
        self.description = description
        self.tile_manager = TileManager(width, height, name)
        self.agents = []
        self.date = datetime.now().strftime("%A, %B %d") # todo
        self.weather = "Sunny" # todo
        self.is_first_day = True
        self.is_new_day = True

    def add_agent(self, agent):
        if not self.tile_manager.is_within_bounds(*agent.position):
            raise ValueError("Agent position is out of bounds.")
        self.agents.append(agent)

    def get_nearest_npc(self, agent):
        others = [other for other in self.agents if other != agent]
        if not others:
            return None
        return min(
            others,
            key=lambda other: self.calculate_distance(agent.position, other.position),
        )

    def calculate_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_state(self):
        return {
            "date": self.date,
            "time": datetime.now().strftime("%H:%M") # todo, remove or change
        }
