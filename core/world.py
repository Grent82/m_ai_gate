from datetime import datetime, timedelta

from core.tile_manager import TileManager
from core.logger import setup_logger

logger = setup_logger(__name__)

class World:
    def __init__(self, name, description, width, height):
        self.name = name
        self.description = description
        self.tile_manager = TileManager(width, height, name)
        self.agents = []
        # Simulated timekeeping
        self.current_time = datetime.now().replace(second=0, microsecond=0)
        self.date = self.current_time.strftime("%A, %B %d")
        self.weather = "Sunny"  # todo
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
            "time": self.current_time.strftime("%H:%M"),
        }

    # --- Time progression -------------------------------------------------
    def advance_time(self, minutes: int = 1) -> None:
        """Advance the world's simulated time and keep agents in sync."""
        self.current_time += timedelta(minutes=minutes)
        # Keep date in sync as well
        self.date = self.current_time.strftime("%A, %B %d")
        for agent in self.agents:
            agent.short_term_memory.current_time = self.current_time
