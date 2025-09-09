from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Event:
    subject: str
    predicate:  Optional[str] = None
    object: Optional[str] = None
    description: str = ""


@dataclass
class ObjectInteraction:
    description: str = ""
    visual_hint: str = ""
    event: Event = field(default_factory=lambda: Event("", None, None))


@dataclass
class ChatInteraction:
    with_whom: Optional[str] = None
    chat_log: List[List[str]] = field(default_factory=list)
    buffer: Dict[str, int] = field(default_factory=dict)
    end_time: Optional[datetime] = None


@dataclass
class PathPlan:
    path: List[Tuple[int, int]] = field(default_factory=list)
    is_set: bool = False


@dataclass
class AgentAction:
    address: Optional[str] = None
    start_time: Optional[datetime] = None
    duration: Optional[int] = None  # in minutes
    description: Optional[str] = None
    visual_hint: Optional[str] = None
    event: Event = field(default_factory=lambda: Event("", None, None))

    object_interaction: ObjectInteraction = field(default_factory=ObjectInteraction)
    chat: ChatInteraction = field(default_factory=ChatInteraction)
    path: PathPlan = field(default_factory=PathPlan)
    # Optional decomposition of the current action into microtasks (strings)
    subtasks: List[str] = field(default_factory=list)

    def start(self, now: Optional[datetime] = None):
        self.start_time = now or datetime.now()

    def get_end_time(self) -> Optional[datetime]:
        if not self.start_time or self.duration is None:
            return None
        clean_start = self.start_time.replace(second=0, microsecond=0)
        if self.start_time.second > 0:
            clean_start += timedelta(minutes=1)
        return clean_start + timedelta(minutes=self.duration)

    def is_chat_finished(self, current_time: datetime) -> bool:
        """
        Determine whether the current action (chat or otherwise) has finished
        based on either chat end_time or the generic action end time.

        Previously this returned True when address was missing, which caused
        immediate replanning for pathless actions like waiting/observing.
        """
        end_time = self.chat.end_time if self.chat.with_whom else self.get_end_time()
        if end_time is None:
            return False
        return current_time >= end_time

    def summary(self) -> Dict[str, Optional[str]]:
        return {
            "persona": self.event.subject,
            "address": self.address,
            "start_datetime": self.start_time.isoformat() if self.start_time else None,
            "duration": self.duration,
            "description": self.description,
            "visual_hint": self.visual_hint,
        }

    def summary_str(self) -> str:
        if not self.start_time:
            return "No current action."
        return (
            f"[{self.start_time.strftime('%A %B %d -- %H:%M %p')}]\n"
            f"Activity: {self.event.subject} is {self.description}\n"
            f"Address: {self.address}\n"
            f"Duration: {self.duration} min"
        )

    def time_str(self) -> str:
        return self.start_time.strftime("%H:%M %p") if self.start_time else "--:--"
