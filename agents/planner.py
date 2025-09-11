from typing import Dict, List, Optional, Tuple
from jinja2 import Environment, FileSystemLoader
from core.agent_action import Event
from memory.long_term_memory import MemoryNode
from models.embeddings import get_embedding
from models.local_model import LocalModel
from core.agent import Agent
from core.world import World
from core.logger import setup_logger
from interfaces.i_planner import IPlanner
from datetime import datetime, timedelta
import re
from .event_triple_generator import EventTripleGenerator
from .chat_manager import ChatManager
from .identity import IdentityReviser

logger = setup_logger(__name__, log_level="DEBUG")

class ModularPlanner(IPlanner):
    def __init__(self, model: LocalModel, template_path: str = "prompts"):
        self.model = model
        self.env = Environment(loader=FileSystemLoader(template_path))
        self.event_triples = EventTripleGenerator(model, template_path)
        self.chat_manager = ChatManager(model, template_path)
        self.identity = IdentityReviser(model, template_path)

    def plan(self, agent: Agent, world: World, retrieved: Dict[str, Dict[str, List[MemoryNode]]], perceived_nodes: Optional[List[MemoryNode]] = None) -> str:
        logger.info("[Planner] Running full daily plan...")

        if not isinstance(agent, Agent) or not isinstance(world, World):
            logger.error("Invalid input to planner. Expected Agent and World references.")
            return "[Error] Invalid input."

        if agent.short_term_memory.is_new_day: 
            self._long_term_planning(agent, world)

        now = agent.short_term_memory.current_time
        current_action = agent.short_term_memory.action

        # If there is no active action yet (e.g., at simulation start), or
        # the current action has finished, determine a new action.
        if (
            current_action.start_time is None
            or current_action.duration is None
            or not current_action.description
            or current_action.is_chat_finished(now)
        ):
            logger.info("[Planner] Current action expired. Determining new plan...")
            self._determine_next_action(agent, world) # todo

        # Prefer reacting to live percepts first; fall back to retrieved-context event
        focused_event = None
        if perceived_nodes:
            focused_event = self._select_focused_event_from_percepts(agent, perceived_nodes)
        if not focused_event and retrieved and retrieved.keys():
            focused_event = self._select_focused_event(retrieved) # todo

        if focused_event:
            reaction = self._decide_reaction(agent, focused_event)
            if reaction:
                if reaction.strip().lower().startswith("ignore"):
                    logger.debug("[Planner] Reaction was 'ignore'; keeping current plan.")
                else:
                    self._apply_reaction(agent, world, focused_event, reaction)
        
        if current_action.event.predicate != "chat with":
            agent.short_term_memory.action.chat.with_whom = None
            agent.short_term_memory.action.chat.chat_log.clear()
            agent.short_term_memory.action.chat.end_time = None

        for name in list(agent.short_term_memory.action.chat.buffer.keys()):
            if name != agent.short_term_memory.action.chat.with_whom and agent.short_term_memory.action.chat.buffer[name] > 0:
                agent.short_term_memory.action.chat.buffer[name] -= 1

        return current_action.address or "idle"

    
    def _long_term_planning(self, agent: Agent, world: World):
        logger.info("[Planner] Daily long term plan...")
        wake_time = self.get_wake_up_time(agent, world)

        if agent.short_term_memory.is_first_day:
            self.get_day_blocks(agent, world, wake_time)
        else:
            # revise identity/status for the day
            try:
                self.identity.revise(agent)
            except Exception as e:
                logger.warning(f"[Planner] Identity revision failed: {e}")

        self.get_hourly_schedule(agent)
        world_state = world.get_state()
        current_time = world_state["time"]
        schedule_summary = self.get_schedule_summary(agent, current_time)
        logger.debug(schedule_summary)
        #tasks = self.generate_microtasks(agent, world, schedule_summary)
        
        thought = f"This is {agent.name}'s plan for {world.date}:"
        for i in agent.short_term_memory.daily_schedule_hourly:
            thought += f" {i['time']} - {i['task']},"
        thought = thought[:-1] + "."
        embedding = get_embedding(thought)
        relevance = 5
        
        logger.debug(f"Add thought '{thought}' to memory")
        event = Event(agent.name, "plan", agent.short_term_memory.current_time.strftime('%A %B %d'), thought)
        agent.long_term_memory.add_thought(event, relevance, set(["plan"]), None, embedding)

        # Flip flags after planning is complete
        agent.short_term_memory.is_first_day = False
        agent.short_term_memory.is_new_day = False


    def _render_prompt(self, template_name: str, context: dict) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)

    def get_wake_up_time(self, agent: Agent, world: World) -> str:
        logger.info("[Planner] Determining wake-up time...")
        context = {"agent": agent.get_state()}
        prompt = self._render_prompt("wake_up_time.txt", context)
        response = self.model.generate(prompt, max_tokens=50, stop=[">>", "User:", "###"])
        logger.debug(f"Wake-up time response: {response}")
        return response.strip()

    def get_day_blocks(self, agent: Agent, world: World, wake_time: str) -> None:
        logger.info("[Planner] Generating day block plan...")
        context = {
            "agent": agent.get_state(),
            "wake_time": wake_time
        }
        prompt = self._render_prompt("day_blocks.txt", context)
        response = self.model.generate(prompt, max_tokens=300, stop=["</day_plan>", "User:", "###"])
        logger.debug(f"Day block response: {response}")
        agent.short_term_memory.daily_schedule = self.parse_day_blocks(response.strip())

    def get_hourly_schedule(self, agent: Agent) -> None:
        logger.info("[Planner] Generating hourly plan from day blocks...")
        schedule: Dict[str, str] = {}

        for block in agent.short_term_memory.daily_schedule:
            try:
                start_time = datetime.strptime(block["start"], "%H:%M")
                end_time = datetime.strptime(block["end"], "%H:%M")
                description = block["description"]

                for hour in range(24):
                    bin_start = datetime.strptime(f"{hour:02}:00", "%H:%M")
                    bin_end = bin_start + timedelta(hours=1)
                    if bin_end > start_time and bin_start < end_time:
                        schedule[bin_start.strftime("%H:%M")] = description
            except ValueError as e:
                logger.error(f"Error parsing time block: {block} - {e}")

        # Fill in any missing hours with "Sleeping"
        for hour in range(24):
            hour_str = f"{hour:02}:00"
            if hour_str not in schedule:
                schedule[hour_str] = "Sleeping"

        final_schedule = [
            {"time": t, "task": schedule[t]}
            for t in sorted(schedule.keys(), key=lambda x: datetime.strptime(x, "%H:%M"))
        ]
        logger.debug(f"Hourly schedule from daily blocks: {final_schedule}")
        agent.short_term_memory.daily_schedule_hourly = final_schedule

    def generate_microtasks(self, agent: Agent, world: World, schedule_summary: str) -> str:
        logger.info(f"[Planner] Generating microtasks from schedule summary...")
        logger.debug(f"[Planner] Schedule summary '{schedule_summary}'")
        context = {
            "agent": agent.get_state(),
            "world": world.get_state(),
            "schedule_summary": schedule_summary,
        }
        prompt = self._render_prompt("microtasks.txt", context)
        logger.debug(f"[Planner] Microtasks prompt =>\n{prompt}")
        response = self.model.generate(prompt, max_tokens=500, stop=["</microtasks>", "User:", "###"])
        logger.debug(f"Microtasks response: {response}")
        return response.strip()

    def get_tile_actions(self, world: World, agent: Agent) -> str:
        logger.info(f"[Planner] Generating actions for tile...")
        location = world.tile_manager.get_nearby_tiles_positions(agent.position)
        context = {
            "agent": agent.get_state(),
            "tile": location
        }
        prompt = self._render_prompt("tile_actions.txt", context)
        response = self.model.generate(prompt, max_tokens=150)
        logger.debug(f"Tile actions response: {response}")
        return response.strip()

    def parse_hourly_schedule(self, schedule_text: str) -> List[Dict[str, str]]:
        schedule = []
        lines = schedule_text.strip().split("\n")
        for line in lines:
            if " - " in line:
                time_part, task_part = line.split(" - ", 1)
                schedule.append({"time": time_part.strip(), "task": task_part.strip()})
        logger.debug(f"Hourly Schedule: {schedule}")
        return schedule

    def parse_day_blocks(self, block_text: str) -> List[Dict[str, str]]:
        blocks: List[Dict[str, str]] = []
        if not block_text:
            logger.warning("[Planner] Empty day block text provided.")
            return blocks

        lines = block_text.strip().split("\n")

        # Pattern that tolerates different dashes and odd minute suffixes (e.g., '21:0ayer')
        # Example expected line: "05:30–06:30: Do something"
        line_pattern = re.compile(r"^\s*([0-2]?\d\s*:\s*\S+)\s*[–—-]\s*([0-2]?\d\s*:\s*\S+)\s*:\s*(.+)$")

        def sanitize_time_token(token: str) -> Optional[str]:
            token = token.strip()
            # Capture hours and up to two minute digits, ignoring trailing junk like 'ayer'
            m = re.match(r"^(\d{1,2})\s*:\s*([0-5]?\d)", token)
            if not m:
                return None
            hh_str, mm_str = m.groups()
            try:
                hh = int(hh_str)
            except ValueError:
                return None
            if not (0 <= hh <= 23):
                return None
            # If minutes is a single digit (e.g., '5' or '0'), interpret as '05' / '00'
            if len(mm_str) == 1:
                mm_str = mm_str.zfill(2)
            try:
                mm = int(mm_str)
            except ValueError:
                return None
            if not (0 <= mm <= 59):
                return None
            return f"{hh:02d}:{mm:02d}"

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            # Normalize list bullets and dashes
            line = re.sub(r"^[•\-*\u2022]+\s*", "", line)
            line = line.replace("—", "-").replace("–", "-").replace("−", "-")

            m = line_pattern.match(line)
            if not m:
                logger.warning(f"[Planner] Malformed day block line (format): '{raw_line}'")
                continue

            start_tok, end_tok, description = m.groups()
            start = sanitize_time_token(start_tok)
            end = sanitize_time_token(end_tok)

            if not start or not end:
                logger.warning(
                    f"[Planner] Malformed time in day block line; skipping: '{raw_line}' (start='{start_tok}', end='{end_tok}')"
                )
                continue

            try:
                start_dt = datetime.strptime(start, "%H:%M")
                end_dt = datetime.strptime(end, "%H:%M")
            except ValueError:
                logger.warning(f"[Planner] Unparseable times in line; skipping: '{raw_line}' -> start='{start}', end='{end}'")
                continue

            # Handle cross‑midnight ranges by splitting into two blocks
            if end_dt <= start_dt:
                if start != end:
                    logger.debug(
                        f"[Planner] Splitting cross-midnight block: '{raw_line}' -> ['{start}-23:59', '00:00-{end}']"
                    )
                    blocks.append({
                        "start": start,
                        "end": "23:59",
                        "description": description.strip(),
                    })
                    blocks.append({
                        "start": "00:00",
                        "end": end,
                        "description": description.strip(),
                    })
                else:
                    logger.warning(
                        f"[Planner] Zero-length block; skipping: '{raw_line}' -> start='{start}', end='{end}'"
                    )
                continue

            blocks.append({
                "start": start,
                "end": end,
                "description": description.strip()
            })

        if not blocks:
            logger.warning("[Planner] No valid day blocks parsed from model output.")

        logger.debug(f"Parsed Day Blocks: {blocks}")
        return blocks

    def get_schedule_summary(self, agent: Agent, current_time: str) -> str:
        try:
            now = datetime.strptime(current_time, "%H:%M")

            hourly_schedule = agent.short_term_memory.daily_schedule_hourly

            # Find the schedule entry that contains 'now'
            idx = None
            for i, entry in enumerate(hourly_schedule):
                entry_time = datetime.strptime(entry["time"], "%H:%M")
                next_time = entry_time + timedelta(hours=1)
                if entry_time <= now < next_time:
                    idx = i
                    break

            if idx is None:
                return f"No known activity for {agent.name} before {current_time}."

            task = hourly_schedule[idx]["task"]

            # Walk backward to find the start of this contiguous task block
            start_idx = idx
            while start_idx - 1 >= 0 and hourly_schedule[start_idx - 1]["task"] == task:
                start_idx -= 1
            start_time = hourly_schedule[start_idx]["time"]

            # Walk forward to find when this task changes next
            end_idx = idx + 1
            while end_idx < len(hourly_schedule) and hourly_schedule[end_idx]["task"] == task:
                end_idx += 1
            end_time = hourly_schedule[end_idx]["time"] if end_idx < len(hourly_schedule) else "end of day"

            return f"From {start_time} to {end_time}, {agent.name} is planning on {task}."

        except Exception as e:
            logger.error(f"Error generating schedule summary: {e}")
            return "No valid schedule summary available."

    def _determine_next_action(self, agent: Agent, world: World):
        logger.info("[Planner] Determining next action from current plan...")

        now = agent.short_term_memory.current_time
        time_key = now.strftime("%H:%M")
        current_task_desc, remaining_minutes = self._get_current_task_from_schedule(agent)

        available_sectors = agent.spatial_memory.get_accessible_sectors(world.name)
        if not available_sectors:
            try:
                tm = world.tile_manager
                seen = set()
                for x in range(tm.width):
                    for y in range(tm.height):
                        t = tm.get_tile(x, y)
                        if t.sector:
                            seen.add(t.sector)
                available_sectors = sorted(seen)
                logger.debug("[Planner] Fallback sectors from world: %s", available_sectors)
            except Exception as e:
                logger.debug("[Planner] Could not build fallback sector list: %s", e)

        current_sector = world.tile_manager.get_current_sector(agent.position)
        sector_prompt_context = {
            "agent": agent.get_state(),
            "task_description": current_task_desc,
            "available_sectors": available_sectors,
            "current_sector": current_sector,
        }
        sector_prompt = self._render_prompt("action_sector.txt", sector_prompt_context)
        logger.debug(f"[Planner] Sector prompt =>\n{sector_prompt}")
        raw_sector = self.model.generate(
            sector_prompt,
            max_tokens=32,
            stop=["User:", "###"],
            temperature=0.1,
            top_p=0.5,
            allowed_strings=available_sectors,
        ).strip()
        logger.debug(
            "[Planner] Sector raw output='%s', options=%s, current='%s'",
            raw_sector,
            sector_prompt_context["available_sectors"],
            sector_prompt_context["current_sector"],
        )
        sector = self._pick_from_options(
            raw_sector,
            available_sectors,
            current_sector,
        )

        available_arenas = agent.spatial_memory.get_accessible_arenas(world.name, sector)
        if not available_arenas:
            try:
                tm = world.tile_manager
                poss = tm.find_positions(sector=sector, include_collidable=True)
                seen = []
                seen_set = set()
                for (x, y) in poss:
                    a = tm.get_tile(x, y).arena
                    if a and a not in seen_set:
                        seen.append(a)
                        seen_set.add(a)
                available_arenas = seen
                logger.debug("[Planner] Fallback arenas for sector '%s': %s", sector, available_arenas)
            except Exception as e:
                logger.debug("[Planner] Could not build fallback arenas: %s", e)

        current_arena = world.tile_manager.get_tile_path(agent.position, "arena").split(":")[-1]
        default_arena = (
            current_arena if (sector == current_sector and current_arena in available_arenas)
            else (available_arenas[0] if available_arenas else current_arena)
        )

        arena_prompt_context = {
            "agent": agent.get_state(),
            "sector": sector,
            "available_arenas": available_arenas,
            "task_description": current_task_desc
        }
        arena_prompt = self._render_prompt("action_arena.txt", arena_prompt_context)
        logger.debug(f"[Planner] Arena prompt =>\n{arena_prompt}")
        raw_arena = self.model.generate(
            arena_prompt,
            max_tokens=32,
            stop=["User:", "###"],
            temperature=0.1,
            top_p=0.5,
            allowed_strings=available_arenas,
        ).strip()
        logger.debug(
            "[Planner] Arena raw output='%s', options=%s",
            raw_arena,
            available_arenas,
        )
        arena = self._pick_from_options(
            raw_arena,
            available_arenas,
            default_arena,
        )

        available_objects = agent.spatial_memory.get_game_objects_in_arena(world.name, sector, arena)
        if not available_objects:
            try:
                tm = world.tile_manager
                poss = tm.find_positions(sector=sector, arena=arena, include_collidable=True)
                seen = []
                seen_set = set()
                for (x, y) in poss:
                    g = tm.get_tile(x, y).game_object
                    if g and g not in seen_set:
                        seen.append(g)
                        seen_set.add(g)
                available_objects = seen
                logger.debug(
                    "[Planner] Fallback objects for sector='%s', arena='%s': %s",
                    sector,
                    arena,
                    available_objects,
                )
            except Exception as e:
                logger.debug("[Planner] Could not build fallback objects: %s", e)

        structural = {"floor", "wall", "door"}
        default_object = None
        if sector == current_sector and arena == current_arena:
            try:
                current_object = world.tile_manager.get_tile_path(agent.position, "game_object").split(":")[-1]
                if current_object in available_objects:
                    default_object = current_object
            except Exception:
                pass
        if not default_object:
            default_object = next((o for o in available_objects if o not in structural), None)
        if not default_object:
            default_object = available_objects[0] if available_objects else "floor"

        # Improve options fed to the LLM by filtering structural items when possible
        non_structural_objects = [o for o in available_objects if o not in structural]
        object_options_for_prompt = non_structural_objects or available_objects
        logger.debug(
            "[Planner] Object options (filtered=%s): %s",
            bool(non_structural_objects),
            object_options_for_prompt,
        )

        object_prompt_context = {
            "agent": agent.get_state(),
            "arena": arena,
            "available_objects": object_options_for_prompt,
            "task_description": current_task_desc
        }
        object_prompt = self._render_prompt("action_game_object.txt", object_prompt_context)
        logger.debug(f"[Planner] Object prompt =>\n{object_prompt}")
        raw_object = self.model.generate(
            object_prompt,
            max_tokens=32,
            stop=["User:", "###"],
            temperature=0.1,
            top_p=0.5,
            allowed_strings=object_options_for_prompt,
        ).strip()
        logger.debug(
            "[Planner] Object raw output='%s', options=%s",
            raw_object,
            object_options_for_prompt,
        )
        game_object = self._pick_from_options(
            raw_object,
            object_options_for_prompt,
            default_object,
        )

        if available_arenas and arena not in available_arenas:
            logger.debug("[Planner] Adjusted arena '%s' not in available list, using default '%s'", arena, default_arena)
            arena = default_arena
        if available_objects and game_object not in available_objects:
            logger.debug("[Planner] Adjusted object '%s' not in available list, using default '%s'", game_object, default_object)
            game_object = default_object

        address = f"{world.name}:{sector}:{arena}:{game_object}"

        duration = min(60, max(5, remaining_minutes))  # cap and floor to reasonable bounds

        try:
            triples = self.event_triples.generate_event(agent, current_task_desc)
            if triples:
                s, p, o = triples[0]
                event = Event(s, p, o, current_task_desc)
            else:
                event = Event(agent.name, "is", current_task_desc, current_task_desc)
        except Exception as e:
            logger.warning(f"[Planner] Event triple generation failed: {e}")
            event = Event(agent.name, "is", current_task_desc, current_task_desc)

        agent.short_term_memory.action.description = current_task_desc
        agent.short_term_memory.action.event = event
        agent.short_term_memory.action.address = address
        agent.short_term_memory.action.duration = duration
        agent.short_term_memory.action.start(now)

        try:
            obj_desc = self._generate_object_interaction(agent, game_object, current_task_desc)
            agent.short_term_memory.action.object_interaction.description = obj_desc
            obj_triples = self.event_triples.generate_event(agent, obj_desc)
            if obj_triples:
                s, p, o = obj_triples[0]
                agent.short_term_memory.action.object_interaction.event = Event(s, p, o, obj_desc)
        except Exception as e:
            logger.debug(f"[Planner] Object interaction generation skipped: {e}")

        try:
            summary = f"From {time_key} for {duration} minutes, {agent.name} is {current_task_desc}."
            micro = self.generate_microtasks(agent, world, summary)
            agent.short_term_memory.action.subtasks = self._parse_microtasks_text(micro)
        except Exception as e:
            logger.debug(f"[Planner] Microtasks not generated: {e}")

        logger.debug(
            "[Planner] Chosen: sector='%s', arena='%s', object='%s' -> address='%s'",
            sector,
            arena,
            game_object,
            address,
        )
        logger.info(f"[Planner] Planned action: '{current_task_desc}' at {address} for {duration}min.")

    def _pick_from_options(self, raw: str, options: List[str], default: str) -> str:
        """
        Safely pick one of the provided options from a raw model output.
        - Case-insensitive matching
        - Accepts outputs containing extra text or JSON-like wrappers
        - Falls back to default if no match found
        """
        if not options:
            logger.debug("[Planner] No options provided. Using default='%s' for raw='%s'", default, raw)
            return default or "here"

        options_sorted = sorted(options, key=len, reverse=True)
        raw_norm = raw.strip().strip('{}[]()"').strip()

        for opt in options:
            if raw_norm.lower() == opt.lower():
                logger.debug("[Planner] Exact match for raw='%s' -> '%s'", raw, opt)
                return opt

        for opt in options_sorted:
            if opt.lower() in raw.lower():
                logger.debug("[Planner] Substring match for raw='%s' -> '%s'", raw, opt)
                return opt

        chosen = default or (options_sorted[-1] if options_sorted else "here")
        logger.debug(
            "[Planner] No match found for raw='%s'. Falling back to '%s' (default). Options=%s",
            raw,
            chosen,
            options,
        )
        return chosen


    def _get_current_task_from_schedule(self, agent: Agent) -> Tuple[str, int]:
        """
        Determine the current task and remaining duration using the hourly schedule
        (which already accounts for gaps and cross‑midnight sleep filling).
        Fallback to day blocks only if hourly schedule is missing.
        """
        now = agent.short_term_memory.current_time
        hourly = agent.short_term_memory.daily_schedule_hourly or []
        try:
            if hourly:
                idx = None
                for i, entry in enumerate(hourly):
                    entry_time = datetime.strptime(entry["time"], "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day
                    )
                    next_time = entry_time + timedelta(hours=1)
                    if entry_time <= now < next_time:
                        idx = i
                        break
                if idx is None:
                    return "idle", 15

                task = hourly[idx]["task"]
                end_idx = idx + 1
                while end_idx < len(hourly) and hourly[end_idx]["task"] == task:
                    end_idx += 1
                end_time = datetime.strptime(hourly[end_idx - 1]["time"], "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                ) + timedelta(hours=1)
                remaining = max(1, int((end_time - now).total_seconds() // 60))
                return task, remaining

            for block in agent.short_term_memory.daily_schedule:
                start = datetime.strptime(block["start"], "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                )
                end = datetime.strptime(block["end"], "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                )
                if start <= now < end:
                    duration = int((end - now).total_seconds() // 60)
                    return block["description"], duration
        except Exception:
            pass
        return "idle", 30

    def _select_focused_event(self, retrieved: Dict[str, Dict[str, List[MemoryNode]]]) -> Optional[Dict[str, List[MemoryNode]]]:
        logger.info("[Planner] Selecting most relevant focused event...")

        if not retrieved:
            logger.debug("[Planner] Retrieved is empty.")
            return None

        def avg_relevance(bundle: Dict[str, List[MemoryNode]]) -> float:
            nodes = bundle.get("context_nodes", [])
            if not nodes:
                return 0.0
            return sum(n.relevance for n in nodes) / len(nodes)

        best_key = None
        best_score = -1.0
        best_bundle = None

        for key, bundle in retrieved.items():
            if "current_event" not in bundle or not isinstance(bundle["current_event"], Event):
                continue

            score = avg_relevance(bundle)
            logger.debug(f"[Planner] Bundle '{key}' has avg relevance: {score:.3f}")

            if score > best_score:
                best_score = score
                best_key = key
                best_bundle = bundle

        if best_bundle:
            logger.info(f"[Planner] Selected focused event from bundle '{best_key}' with relevance {best_score:.3f}")
        else:
            logger.debug("[Planner] No suitable focused event found.")

        return best_bundle

    def _select_focused_event_from_percepts(self, agent: Agent, perceived_nodes: List[MemoryNode]) -> Optional[Dict[str, List[MemoryNode]]]:
        """Pick a live perceived event (prefer non-self subjects) to consider for reaction."""
        try:
            events: List[Event] = []
            for n in perceived_nodes:
                if getattr(n, "node_type", "") != "event":
                    continue
                ev = getattr(n, "event", None)
                if not ev or not ev.subject:
                    continue
                if ev.subject == agent.name:
                    continue
                events.append(ev)

            if not events:
                return None

            ev = events[0]
            logger.info("[Planner] Selected focused live event: %s %s %s", ev.subject, ev.predicate or "is", ev.object or "")
            return {"current_event": ev, "context_nodes": []}
        except Exception as e:
            logger.debug("[Planner] Live percept selection failed: %s", e)
            return None



    def _decide_reaction(self, agent: Agent, context_bundle: Dict[str, List[MemoryNode]]) -> Optional[str]:
        logger.info("[Planner] Deciding whether to react to an event...")
        event = context_bundle["current_event"]
        memories = [n.event.description for n in context_bundle.get("context_nodes", [])]

        now = agent.short_term_memory.current_time
        current_desc = (agent.short_term_memory.action.description or "").lower()
        if now and now.hour >= 23:
            return None
        if "sleep" in current_desc or "in bed" in current_desc:
            return None
        if agent.short_term_memory.action.chat.with_whom:
            return None

        context = {
            "agent": agent.get_state(),
            "event": {
                "subject": event.subject,
                "predicate": event.predicate or "is",
                "object": event.object or "someone",
                "description": event.description
            },
            "memories": memories
        }

        prompt = self._render_prompt("should_react_to_event.txt", context)
        logger.debug(f"[Planner] Reaction prompt =>\n{prompt}")
        response = self.model.generate(
            prompt,
            max_tokens=32,
            stop=["</reaction>", "User:", "###"],
            temperature=0.2,
            top_p=0.6,
        )
        reaction = response.strip()
        logger.debug(f"[Planner] Reaction decision: {reaction}")
        return reaction if reaction else None

    def _apply_reaction(self, agent: Agent, world: World, context_bundle: Dict[str, List[MemoryNode]], reaction_mode: str):
        logger.info(f"[Planner] Applying reaction: {reaction_mode}")
        event = context_bundle["current_event"]
        now = agent.short_term_memory.current_time
        try:
            x, y = agent.position
            world.tile_manager.remove_event_from_tile(x, y, agent.short_term_memory.action.event)
        except Exception:
            pass

        if reaction_mode.startswith("chat with"):
            # Prefer explicit target from the reaction text; otherwise use the event subject
            target_text = reaction_mode[len("chat with"):].strip()
            target = target_text or event.subject or event.object or "someone"
            convo = self.chat_manager.generate_conversation(agent, target)
            summary = self.chat_manager.summarize_conversation(agent, convo)
            turns = max(1, len(convo))
            minutes = max(5, min(20, (turns + 1) // 2))

            agent.short_term_memory.action.chat.with_whom = target
            agent.short_term_memory.action.chat.chat_log = [[s, u] for s, u in convo]

            # Do not start the timer yet; executor will start it upon arrival
            agent.short_term_memory.action.description = f"heading to chat with {target}"
            agent.short_term_memory.action.event = Event(agent.name, "chat with", target, summary)
            agent.short_term_memory.action.duration = minutes
            agent.short_term_memory.action.start(now)

        elif reaction_mode.startswith("wait"):
            agent.short_term_memory.action.description = "waiting quietly"
            agent.short_term_memory.action.event = Event(agent.name, "is", "waiting", "The agent decides to wait quietly.")
            agent.short_term_memory.action.duration = 10
            agent.short_term_memory.action.start(now)

        elif reaction_mode.startswith("move to"):
            location = reaction_mode[len("move to "):].strip() or "somewhere"
            agent.short_term_memory.action.description = f"moving to {location}"
            agent.short_term_memory.action.event = Event(agent.name, "move to", location, f"{agent.name} heads toward {location}.")
            agent.short_term_memory.action.duration = 15
            agent.short_term_memory.action.address = location
            agent.short_term_memory.action.start(now)

        elif reaction_mode.startswith("observe"):
            target = event.object or "surroundings"
            agent.short_term_memory.action.description = f"observing {target}"
            agent.short_term_memory.action.event = Event(agent.name, "observe", target, f"{agent.name} watches {target} closely.")
            agent.short_term_memory.action.duration = 5
            agent.short_term_memory.action.start(now)

        elif reaction_mode.startswith("help"):
            target = event.object or "someone"
            agent.short_term_memory.action.description = f"helping {target}"
            agent.short_term_memory.action.event = Event(agent.name, "help", target, f"{agent.name} offers help to {target}.")
            agent.short_term_memory.action.duration = 10
            agent.short_term_memory.action.start(now)

        elif reaction_mode.startswith("follow"):
            target = event.object or "someone"
            agent.short_term_memory.action.description = f"following {target}"
            agent.short_term_memory.action.event = Event(agent.name, "follow", target, f"{agent.name} quietly follows {target}.")
            agent.short_term_memory.action.duration = 10
            agent.short_term_memory.action.start(now)

        else:
            logger.info(f"[Planner] No handler for reaction: {reaction_mode}. Keeping current plan.")
            return

    def _generate_object_interaction(self, agent: Agent, game_object: str, task_description: str) -> str:
        context = {
            "agent": agent.get_state(),
            "game_object": game_object,
            "task_description": task_description,
        }
        prompt = self._render_prompt("object_interaction.txt", context)
        response = self.model.generate(prompt, max_tokens=80, stop=["\n", "</obj>", "###"]).strip()
        return response

    def _parse_microtasks_text(self, text: str) -> List[str]:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        cleaned: List[str] = []
        for l in lines:
            s = l
            # Drop bullets/numbers
            s = re.sub(r"^[\u2022•\-\*\d\)\.\s]+", "", s)
            # Remove time ranges like "HH:MM - HH:MM -" or single "HH:MM:"
            s = re.sub(r"^(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:-\s*\d{1,2}:\d{2}(?::\d{2})?)?\s*[-:]?\s*)+", "", s)
            # Remove stray leading ":MM:" leftovers
            s = re.sub(r"^:\d{1,2}:\s*", "", s)
            # Normalize whitespace
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                cleaned.append(s)
        return cleaned
