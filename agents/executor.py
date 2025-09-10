from collections import deque
import heapq
from datetime import timedelta
from typing import List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from core.agent import Agent
from core.agent_action import Event
from core.world import World
from memory.long_term_memory import MemoryNode
from models.embeddings import get_embedding
from models.local_model import LocalModel
from core.logger import setup_logger
from interfaces.i_executor import IExecutor

logger = setup_logger(__name__, log_level="DEBUG")


class Executor(IExecutor):
    """
    Executes the current planned action for an agent.

    Responsibilities:
    - Plan a path to the target address (if not already set)
    - Step the agent along the path (bounded by max_steps)
    - Attach the current action's event to the agent's tile so it can be perceived
    - Optionally convert completed subtasks into long-term memories

    The design mirrors other agents modules: template-driven, LocalModel-friendly,
    but keeps execution logic deterministic and local.
    """

    def __init__(self, model: LocalModel, template_path: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(template_path))

    # Public API --------------------------------------------------------
    def execute(self, agent: Agent, world: World, max_steps: int = 1) -> str:
        """
        Advance the agent along its current plan.

        - Computes a path if needed
        - Moves up to max_steps along the path
        - Places the action's event on the current tile
        - Returns a concise status string
        """
        agent.short_term_memory.current_time = world.current_time
        action = agent.short_term_memory.action

        if not action:
            logger.debug("[Executor] No active action; agent is idle.")
            return "idle"

        self._attach_event_to_current_tile(agent, world)

        if action.event and (action.event.object or "").lower() == "waiting":
            return "waiting"

        if not action.path.is_set or not action.path.path:
            target = self._select_target_tile(agent, world)
            if target is None:
                logger.warning(
                    "[Executor] No reachable target found for address '%s'.", action.address
                )
                return "no target"
            path = self._find_path(world, agent.position, target)
            if len(path) <= 1:
                action.path.path = []
                action.path.is_set = True
            else:
                action.path.path = path[1:]
                action.path.is_set = True
            logger.debug("[Executor] Planned path len=%d", len(action.path.path))

        steps_taken = 0
        while steps_taken < max_steps and action.path.path:
            nx, ny = action.path.path.pop(0)
            prev = agent.position
            agent.position = (nx, ny)
            logger.debug("[Executor] Moved from %s to %s", prev, agent.position)
            self._move_event_between_tiles(world, prev, agent.position, action.event)
            steps_taken += 1

        if not action.path.path:
            self._on_arrival(agent, world)

        if action.event and (action.event.predicate or "").lower() == "chat with":
            if action.chat.end_time and world.current_time < action.chat.end_time:
                return f"chatting with {action.chat.with_whom or 'someone'}"
            return f"moving to {action.event.object or 'someone'} to chat"

        return f"moved {steps_taken} step(s)" if steps_taken else "at destination"

    # Internal helpers --------------------------------------------------
    def _attach_event_to_current_tile(self, agent: Agent, world: World) -> None:
        try:
            x, y = agent.position
            t = world.tile_manager.get_tile(x, y)
            if agent.short_term_memory.action.event not in t.events:
                world.tile_manager.add_event_to_tile(x, y, agent.short_term_memory.action.event)
                logger.debug(
                    "[Executor] Attached event to current tile: %s", agent.short_term_memory.action.event
                )
        except Exception as e:
            logger.debug("[Executor] Could not attach event to current tile: %s", e)

    def _move_event_between_tiles(
        self, world: World, prev: Tuple[int, int], curr: Tuple[int, int], event: Event
    ) -> None:
        try:
            world.tile_manager.remove_event_from_tile(prev[0], prev[1], event)
        except Exception:
            pass
        try:
            world.tile_manager.add_event_to_tile(curr[0], curr[1], event)
        except Exception as e:
            logger.debug("[Executor] Failed to add event on move: %s", e)

    def _select_target_tile(self, agent: Agent, world: World) -> Optional[Tuple[int, int]]:
        """
        Choose the closest non-collidable tile that matches the action address.

        If no exact match exists, degrade gracefully by dropping the game_object,
        then the arena.
        """
        address = agent.short_term_memory.action.address or ""
        tm = world.tile_manager

        if address and "<waiting>" in address:
            # Format: "<waiting> x y" or "... <waiting> x y"
            try:
                parts = address.split()
                if "<waiting>" in parts:
                    i = parts.index("<waiting>")
                    x = int(parts[i + 1])
                    y = int(parts[i + 2])
                    if tm.is_within_bounds(x, y) and not tm.is_collidable(x, y):
                        return (x, y)
            except Exception:
                logger.debug("[Executor] Failed to parse <waiting> coordinates from '%s'", address)
            return agent.position

        if address and "<random>" in address:
            import random
            nearby = tm.get_nearby_tiles_positions(agent.position, radius=agent.vision_range)
            options = [(x, y) for (x, y) in nearby if not tm.is_collidable(x, y)]
            if options:
                return random.choice(options)
            return agent.position

        event = agent.short_term_memory.action.event
        if event and event.object:
            for other in world.agents:
                if other is agent:
                    continue
                if other.name == event.object:
                    return other.position

        if address and ":" not in address:
            import random
            nearby = tm.get_nearby_tiles_positions(agent.position, radius=agent.vision_range)
            options = [(x, y) for (x, y) in nearby if not tm.is_collidable(x, y)]
            if options:
                return random.choice(options)

        candidates = tm.find_positions_by_address(address)
        if not candidates:
            parts = [p.strip() for p in address.split(":")]
            sector = parts[1] if len(parts) > 1 else None
            arena = parts[2] if len(parts) > 2 else None
            obj = parts[3] if len(parts) > 3 else None
            if sector or arena or obj:
                try:
                    candidates = tm.find_positions(sector=sector, arena=arena, game_object=obj, include_collidable=True)
                except Exception:
                    candidates = []
            if not candidates:
                if sector and arena:
                    candidates = tm.find_positions(sector=sector, arena=arena)
                if not candidates and sector:
                    candidates = tm.find_positions(sector=sector)

        if not candidates:
            return None

        ax, ay = agent.position
        best = min(candidates, key=lambda p: (p[0] - ax) ** 2 + (p[1] - ay) ** 2)

        bx, by = best
        try:
            if tm.is_collidable(bx, by):
                target_tile = tm.get_tile(bx, by)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = bx + dx, by + dy
                    if not tm.is_within_bounds(nx, ny):
                        continue
                    if tm.is_collidable(nx, ny):
                        continue
                    t2 = tm.get_tile(nx, ny)
                    if t2.sector == target_tile.sector and t2.arena == target_tile.arena:
                        return (nx, ny)
                return best
        except Exception:
            pass

        return best

    def _find_path(
        self, world: World, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Cost-aware pathfinding (Dijkstra) that prefers low-cost terrain like gravel.

        Returns the list of coordinates from start to goal (inclusive).
        """
        if start == goal:
            return [start]

        tm = world.tile_manager

        def neighbors(x: int, y: int):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not tm.is_within_bounds(nx, ny):
                    continue
                if tm.is_collidable(nx, ny):
                    continue
                yield (nx, ny)

        def move_cost(x: int, y: int) -> float:
            try:
                obj = tm.get_tile(x, y).game_object or ""
            except Exception:
                return 1.0
            obj = obj.lower()
            if obj in {"gravel", "path", "road"}:
                return 0.5
            if obj in {"bush"}:
                return 1.2
            return 1.0

        pq: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(pq, (0.0, start))
        came_from: dict[Tuple[int, int], Tuple[int, int] | None] = {start: None}
        g_cost: dict[Tuple[int, int], float] = {start: 0.0}

        while pq:
            cost, (cx, cy) = heapq.heappop(pq)
            if (cx, cy) == goal:
                break
            if cost > g_cost.get((cx, cy), float("inf")):
                continue
            for nx, ny in neighbors(cx, cy):
                step = move_cost(nx, ny)
                new_cost = cost + step
                if new_cost < g_cost.get((nx, ny), float("inf")):
                    g_cost[(nx, ny)] = new_cost
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(pq, (new_cost, (nx, ny)))

        if goal not in came_from:
            logger.debug("[Executor] No path found from %s to %s", start, goal)
            return [start]

        path: List[Tuple[int, int]] = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    def _on_arrival(self, agent: Agent, world: World) -> None:
        """
        Handle arrival at destination:
        - Execute one microtask (if any), storing it as an event in long-term memory
        - Ensure the action event is present on the destination tile
        """
        action = agent.short_term_memory.action
        self._attach_event_to_current_tile(agent, world)

        if action.event and (action.event.predicate or "").lower() == "chat with" and not action.chat.end_time:
            target = action.event.object or "someone"
            if not action.chat.with_whom:
                action.chat.with_whom = target
            minutes = max(5, action.duration or 10)
            action.chat.end_time = world.current_time + timedelta(minutes=minutes)
            action.chat.buffer[target] = max(5, minutes // 2)
            action.description = f"chatting with {target}"
            logger.debug(
                "[Executor] Started chat with '%s' for %d minutes at %s",
                target,
                minutes,
                world.current_time,
            )

        if action.subtasks:
            sub = action.subtasks.pop(0)
            desc = f"{agent.name} {sub}."
            evt = Event(agent.name, "does", sub, desc)
            embedding = get_embedding(desc)
            try:
                agent.long_term_memory.add_event(evt, relevance=5.0, keywords={agent.name, "does", sub}, embedding=embedding)
                logger.debug("[Executor] Logged subtask as event: %s", desc)
            except Exception as e:
                logger.debug("[Executor] Failed to log subtask memory: %s", e)

    # Lifecycle utilities -----------------------------------------------
    def cleanup_expired(self, agent: Agent, world: World) -> Optional[str]:
        """
        If the current action is finished at the world's time, remove its event from the tile
        and reset path flags. Return a message if cleanup occurred.
        """
        now = world.current_time
        action = agent.short_term_memory.action
        if not action:
            return None
        if not action.is_chat_finished(now):
            return None

        try:
            x, y = agent.position
            world.tile_manager.remove_event_from_tile(x, y, action.event)
        except Exception:
            pass

        action.path.path.clear()
        action.path.is_set = False
        action.chat.with_whom = None
        action.chat.chat_log.clear()
        action.chat.end_time = None
        logger.debug("[Executor] Cleaned up expired action at %s", now)
        return "cleaned"
