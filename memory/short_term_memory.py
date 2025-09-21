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
        self.current_time: Optional[datetime] = None
        self.current_tile: Optional[Tuple[int, int]] = None

        self.action = AgentAction()

        # Reflection and control variables
        self.importance_trigger_current: float = 150.0
        self.importance_element_count: int = 0

        self.recency_decay = 0.99
        self.recency_weight = 1
        self.importance_weight = 1
        self.relevance_weight = 1
        self.depth_weight = 0.5

    def add_action(self, action: AgentAction):
        self.action = action

    # --- Persistence ---
    def to_dict(self) -> Dict:
        return {
            "daily_schedule": self.daily_schedule,
            "daily_schedule_hourly": self.daily_schedule_hourly,
            "is_first_day": self.is_first_day,
            "is_new_day": self.is_new_day,
            "current_time": self.current_time.strftime("%B %d, %Y, %H:%M:%S") if self.current_time else None,
            "current_tile": self.current_tile,
            "importance_trigger_current": self.importance_trigger_current,
            "importance_element_count": self.importance_element_count,
            "recency_decay": self.recency_decay,
            "recency_weight": self.recency_weight,
            "importance_weight": self.importance_weight,
            "relevance_weight": self.relevance_weight,
            "depth_weight": self.depth_weight,
            # Action
            "action": {
                "address": self.action.address,
                "start_time": self.action.start_time.strftime("%B %d, %Y, %H:%M:%S") if self.action.start_time else None,
                "duration": self.action.duration,
                "description": self.action.description,
                "visual_hint": self.action.visual_hint,
                "event": [
                    getattr(self.action.event, "subject", None),
                    getattr(self.action.event, "predicate", None),
                    getattr(self.action.event, "object", None),
                    getattr(self.action.event, "description", ""),
                ],
                "chat": {
                    "with_whom": self.action.chat.with_whom,
                    "chat_log": self.action.chat.chat_log,
                    "buffer": self.action.chat.buffer,
                    "end_time": self.action.chat.end_time.strftime("%B %d, %Y, %H:%M:%S") if self.action.chat.end_time else None,
                },
                "object_interaction": {
                    "description": self.action.object_interaction.description,
                    "visual_hint": self.action.object_interaction.visual_hint,
                    "event": [
                        getattr(self.action.object_interaction.event, "subject", None),
                        getattr(self.action.object_interaction.event, "predicate", None),
                        getattr(self.action.object_interaction.event, "object", None),
                        getattr(self.action.object_interaction.event, "description", ""),
                    ],
                },
                "path": {
                    "path": self.action.path.path,
                    "is_set": self.action.path.is_set,
                    "failed": self.action.path.failed,
                },
                "subtasks": self.action.subtasks,
            },
        }

    def save(self, out_path: str) -> None:
        import json, os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, in_path: str) -> "ShortTermMemory":
        import json, os
        from datetime import datetime
        from core.agent_action import AgentAction, Event, ChatInteraction, ObjectInteraction, PathPlan

        stm = cls()
        if not os.path.exists(in_path):
            return stm

        try:
            data = json.load(open(in_path))
        except Exception:
            return stm

        stm.daily_schedule = data.get("daily_schedule", []) or []
        stm.daily_schedule_hourly = data.get("daily_schedule_hourly", []) or []
        stm.is_first_day = bool(data.get("is_first_day", True))
        stm.is_new_day = bool(data.get("is_new_day", True))
        ct = data.get("current_time")
        stm.current_time = datetime.strptime(ct, "%B %d, %Y, %H:%M:%S") if ct else None
        stm.current_tile = tuple(data.get("current_tile")) if data.get("current_tile") else None

        stm.importance_trigger_current = float(data.get("importance_trigger_current", 150.0))
        stm.importance_element_count = int(data.get("importance_element_count", 0))
        stm.recency_decay = float(data.get("recency_decay", 0.99))
        stm.recency_weight = float(data.get("recency_weight", 1.0))
        stm.importance_weight = float(data.get("importance_weight", 1.0))
        stm.relevance_weight = float(data.get("relevance_weight", 1.0))
        stm.depth_weight = float(data.get("depth_weight", 0.5))

        # Action
        a = data.get("action", {}) or {}
        action = AgentAction()
        action.address = a.get("address")
        st = a.get("start_time")
        action.start_time = datetime.strptime(st, "%B %d, %Y, %H:%M:%S") if st else None
        action.duration = a.get("duration")
        action.description = a.get("description")
        action.visual_hint = a.get("visual_hint")
        ev = a.get("event") or [None, None, None, ""]
        action.event = Event(ev[0] or "", ev[1], ev[2], ev[3] or "")

        # Chat
        ch = a.get("chat", {}) or {}
        action.chat.with_whom = ch.get("with_whom")
        action.chat.chat_log = ch.get("chat_log", []) or []
        action.chat.buffer = ch.get("buffer", {}) or {}
        et = ch.get("end_time")
        action.chat.end_time = datetime.strptime(et, "%B %d, %Y, %H:%M:%S") if et else None

        # Object interaction
        oi = a.get("object_interaction", {}) or {}
        action.object_interaction.description = oi.get("description") or ""
        action.object_interaction.visual_hint = oi.get("visual_hint") or ""
        oiev = oi.get("event") or [None, None, None, ""]
        action.object_interaction.event = Event(oiev[0] or "", oiev[1], oiev[2], oiev[3] or "")

        # Path
        pp = a.get("path", {}) or {}
        action.path.path = [tuple(p) for p in (pp.get("path") or [])]
        action.path.is_set = bool(pp.get("is_set", False))
        action.path.failed = bool(pp.get("failed", False))

        action.subtasks = a.get("subtasks", []) or []
        stm.action = action
        return stm
