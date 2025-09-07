from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.agent_action import AgentAction


class ShortTermMemory:
    def __init__(self):
        # Planning and scheduling
        self.daily_schedule: List[Dict[str, str]] = []
        self.daily_schedule_hourly: List[Dict[str, str]] = []
        self.is_first_day = True
        self.is_new_day = True

        # Short-term memory variables
        self.current_time: Optional[datetime] = None  # Tracks the agent's current time for scheduling
        self.current_tile: Optional[Tuple[int, int]] = None

        self.action = AgentAction()

        # Reflection and control variables
        self.importance_trigger_current: float = 150.0
        self.importance_element_count: int = 0

        self.recency_decay = 0.99
        self.recency_weight = 1
        self.importance_weight = 1
        self.relevance_weight = 1

    def add_action(self, action: AgentAction):
        self.action = action

