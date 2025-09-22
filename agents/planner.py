import os
from typing import Dict, List, Optional, Tuple
from jinja2 import Environment, FileSystemLoader
from core.agent_action import Event
from memory.long_term_memory import MemoryNode
from models.embedding_cache import get_embedding_cached
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

        if (
            current_action.start_time is None
            or current_action.duration is None
            or not current_action.description
            or current_action.is_chat_finished(now)
        ):
            logger.info("[Planner] Current action expired. Determining new plan...")
            self._determine_next_action(agent, world) # todo

        focused_event = None
        if perceived_nodes:
            logger.debug(f"[Planner] Perceived nodes count: {len(perceived_nodes)}")
            focused_event = self._select_focused_event_from_percepts(agent, perceived_nodes)

        if not focused_event and retrieved and retrieved.keys():
            logger.debug(f"[Planner] Retrieved memory categories: {list(retrieved.keys())}")
            focused_event = self._select_focused_event(retrieved) # todo

        if focused_event:
            try:
                fe = focused_event.get("current_event") if isinstance(focused_event, dict) else None
                focused_event_description  = getattr(fe, "description", None) or f"{getattr(fe, 'subject', '')} {getattr(fe, 'predicate', '')} {getattr(fe, 'object', '')}".strip()
            except Exception:
                focused_event_description  = "<unknown>"
            logger.info(f"[Planner] Focused event for reaction: {focused_event_description }")
            reaction = self._decide_reaction(agent, world, focused_event)
            if reaction:
                if reaction.strip().lower().startswith("ignore"):
                    logger.debug("[Planner] Reaction was 'ignore'; keeping current plan.")
                else:
                    self._apply_reaction(agent, world, focused_event, reaction)
        
        logger.debug(f"[Planner] Final planned action: {agent.short_term_memory.action}")
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
        embedding = get_embedding_cached(thought)
        relevance = 5
        
        logger.debug(f"Add thought '{thought}' to memory")
        event = Event(agent.name, "plan", agent.short_term_memory.current_time.strftime('%A %B %d'), thought)
        agent.long_term_memory.add_thought(event, relevance, set(["plan"]), None, embedding)

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
        current_sector = world.tile_manager.get_current_sector(agent.position)
        # current sector arenas
        try:
            current_sector_arenas = agent.spatial_memory.get_accessible_arenas(world.name, current_sector) or []
            if not current_sector_arenas:
                tm = world.tile_manager
                poss_cur = tm.find_positions(sector=current_sector, include_collidable=True)
                seen_cur = []
                seen_set_cur = set()
                for (x, y) in poss_cur:
                    a = tm.get_tile(x, y).arena
                    if a and a not in seen_set_cur:
                        seen_cur.append(a)
                        seen_set_cur.add(a)
                current_sector_arenas = seen_cur
        except Exception:
            current_sector_arenas = []

        # living sector arenas
        living_area = getattr(agent, "living_area", "") or ""
        parts = living_area.split(":") if living_area else []
        living_sector = parts[1] if len(parts) >= 2 else ""
        living_sector_arenas: List[str] = []
        if living_sector:
            try:
                tm = world.tile_manager
                poss_l = tm.find_positions(sector=living_sector, include_collidable=True)
                seen_l: List[str] = []
                seen_set_l = set()
                for (x, y) in poss_l:
                    a = tm.get_tile(x, y).arena
                    if a and a not in seen_set_l:
                        seen_l.append(a)
                        seen_set_l.add(a)
                living_sector_arenas = seen_l
            except Exception:
                living_sector_arenas = []

        context = {
            "agent": agent.get_state(),
            "wake_time": wake_time,
            "current_sector_arenas": ", ".join(current_sector_arenas) if current_sector_arenas else "",
            "living_sector_arenas": ", ".join(living_sector_arenas) if living_sector_arenas else "",
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

    def generate_microtasks(
        self,
        agent: Agent,
        world: World,
        *,
        task_description: str,
        start_time: str,
        end_time: str,
        duration_minutes: int,
        sector: str,
        arena: str,
    ) -> str:
        logger.info("[Planner] Generating microtasks for current action window...")
        context = {
            "agent": agent.get_state(),
            "world": world.get_state(),
            "task_description": task_description,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "location": {"sector": sector, "arena": arena},
        }
        prompt = self._render_prompt("microtasks.txt", context)
        logger.debug(f"[Planner] Microtasks prompt =>\n{prompt}")
        response = self.model.generate(
            prompt,
            max_tokens=500,
            stop=["</microtasks>", "User:", "###"],
            temperature=0.3,
            top_p=0.95,
        )
        logger.debug(f"Microtasks response: {response}")
        return response.strip()

    def get_tile_actions(self, world: World, agent: Agent) -> str:
        logger.info(f"[Planner] Generating actions for tile...")
        tm = world.tile_manager
        nearby = tm.get_nearby_tiles_positions(agent.position)
        events_text: List[str] = []
        try:
            for (x, y) in nearby:
                try:
                    t = tm.get_tile(x, y)
                except Exception:
                    continue
                for ev in getattr(t, "events", []) or []:
                    desc = getattr(ev, "description", None)
                    if desc:
                        events_text.append(desc)
                    else:
                        subj = getattr(ev, "subject", "?")
                        pred = getattr(ev, "predicate", "is") or "is"
                        obj = getattr(ev, "object", "") or ""
                        events_text.append(f"{subj} {pred} {obj}".strip())
        except Exception:
            pass

        context = {
            "agent": agent.get_state(),
            "tile": {"events": events_text},
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
            logger.debug(f"Sanitizing time token: '{token}'")
            # Capture hours and up to two minute digits, ignoring trailing junk like 'ayer'
            m = re.match(r"^(\d{1,2})\s*:\s*([0-5]?\d)", token)
            if not m:
                logger.warning(f"Time token format invalid: '{token}'")
                return None
            hh_str, mm_str = m.groups()
            try:
                hh = int(hh_str)
            except ValueError:
                logger.error(f"Hour parsing failed for token '{token}'")
                return None
            if not (0 <= hh <= 23):
                logger.warning(f"Hour out of range in token '{token}'")
                return None
            # If minutes is a single digit (e.g., '5' or '0'), interpret as '05' / '00'
            if len(mm_str) == 1:
                mm_str = mm_str.zfill(2)
            try:
                mm = int(mm_str)
            except ValueError:
                logger.error(f"Minute parsing failed for token '{token}'")
                return None
            if not (0 <= mm <= 59):
                logger.warning(f"Minute out of range in token '{token}'")
                return None
            return f"{hh:02d}:{mm:02d}"

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
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
                logger.warning( f"[Planner] Malformed time in day block line; skipping: '{raw_line}' (start='{start_tok}', end='{end_tok}')" )
                continue

            try:
                start_dt = datetime.strptime(start, "%H:%M")
                end_dt = datetime.strptime(end, "%H:%M")
            except ValueError:
                logger.warning(f"[Planner] Unparseable times in line; skipping: '{raw_line}' -> start='{start}', end='{end}'")
                continue

            if end_dt <= start_dt:
                if start != end:
                    logger.debug( f"[Planner] Splitting cross-midnight block: '{raw_line}' -> ['{start}-23:59', '00:00-{end}']" )
                    blocks.append({ "start": start, "end": "23:59", "description": description.strip(), })
                    blocks.append({ "start": "00:00", "end": end, "description": description.strip(), })
                else:
                    logger.warning( f"[Planner] Zero-length block; skipping: '{raw_line}' -> start='{start}', end='{end}'" )
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

            start_idx = idx
            while start_idx - 1 >= 0 and hourly_schedule[start_idx - 1]["task"] == task:
                start_idx -= 1
            start_time = hourly_schedule[start_idx]["time"]

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

        # Determine Action sector
        current_sector = world.tile_manager.get_current_sector(agent.position)
        try:
            tm = world.tile_manager
            poss_cur = tm.find_positions(sector=current_sector, include_collidable=True)
            seen_cur = []
            seen_set_cur = set()
            for (x, y) in poss_cur:
                a = tm.get_tile(x, y).arena
                if a and a not in seen_set_cur:
                    seen_cur.append(a)
                    seen_set_cur.add(a)
            current_sector_arenas = seen_cur
        except Exception:
            current_sector_arenas = []

        living_area = getattr(agent, "living_area", "") or ""
        living_parts = living_area.split(":") if living_area else []
        living_sector = living_parts[1] if len(living_parts) >= 2 else ""
        living_sector_arenas = []
        if living_sector:
            try:
                tm = world.tile_manager
                poss_l = tm.find_positions(sector=living_sector, include_collidable=True)
                seen_l = []
                seen_set_l = set()
                for (x, y) in poss_l:
                    a = tm.get_tile(x, y).arena
                    if a and a not in seen_set_l:
                        seen_l.append(a)
                        seen_set_l.add(a)
                living_sector_arenas = seen_l
            except Exception as e:
                logger.debug("[Planner] Could not build arenas for living sector '%s': %s", living_sector, e)

        sector_prompt_context = {
            "agent": agent.get_state(),
            "task_description": current_task_desc,
            "available_sectors": available_sectors,
            "agent_living_sector_areas": ", ".join(living_sector_arenas) if living_sector_arenas else "",
            "current_sector": current_sector,
            "current_sector_arenas": ", ".join(current_sector_arenas) if current_sector_arenas else "",
        }
        sector_prompt = self._render_prompt("action_sector_v2.txt", sector_prompt_context)
        logger.debug(f"[Planner] Sector prompt =>\n{sector_prompt}")
        raw_sector = self.model.generate(
            sector_prompt,
            max_tokens=32,
            stop=["User:", "###"],
            temperature=0.25,
            top_p=0.9,
            allowed_strings=available_sectors,
        ).strip()
        logger.debug( "[Planner] Sector raw output='%s', options=%s, current='%s'", raw_sector, sector_prompt_context["available_sectors"], sector_prompt_context["current_sector"], )
        choosen_sector = self._pick_from_options(
            raw_sector,
            available_sectors,
            current_sector,
        )

        # Determine Action arena
        try:
            tm = world.tile_manager
            poss = tm.find_positions(sector=choosen_sector, include_collidable=True)
            seen = []
            seen_set = set()
            for (x, y) in poss:
                a = tm.get_tile(x, y).arena
                if a and a not in seen_set:
                    seen.append(a)
                    seen_set.add(a)
            available_arenas = seen
        except Exception as e:
            logger.debug("[Planner] Could not build arenas for sector '%s': %s", choosen_sector, e)
            available_arenas = []

        # Rebuild current sector arenas for prompt context directly from world
        current_sector_arenas = []
        try:
            tm = world.tile_manager
            poss_cur = tm.find_positions(sector=current_sector, include_collidable=True)
            seen_cur: list[str] = []
            seen_set_cur = set()
            for (x, y) in poss_cur:
                a = tm.get_tile(x, y).arena
                if a and a not in seen_set_cur:
                    seen_cur.append(a)
                    seen_set_cur.add(a)
            current_sector_arenas = seen_cur
        except Exception as e:
            logger.debug("[Planner] Could not build arenas for current sector '%s': %s", current_sector, e)

        current_arena = world.tile_manager.get_tile_path(agent.position, "arena").split(":")[-1]
        
        living_area = getattr(agent, "living_area", "") or ""
        living_parts = living_area.split(":") if living_area else []
        living_sector = living_parts[1] if len(living_parts) >= 2 else ""
        living_sector_arenas = []
        if living_sector:
            try:
                tm = world.tile_manager
                poss_l = tm.find_positions(sector=living_sector, include_collidable=True)
                seen_l = []
                seen_set_l = set()
                for (x, y) in poss_l:
                    a = tm.get_tile(x, y).arena
                    if a and a not in seen_set_l:
                        seen_l.append(a)
                        seen_set_l.add(a)
                living_sector_arenas = seen_l
            except Exception as e:
                logger.debug("[Planner] Could not build arenas for living sector '%s': %s", living_sector, e)

        arena_prompt_context = {
            "agent": agent.get_state(),
            "current_sector": current_sector,
            "choosen_sector": choosen_sector,
            "available_arenas": available_arenas,
            "task_description": current_task_desc,
            "current_sector_arenas": ", ".join(current_sector_arenas) if current_sector_arenas else "",
            "agent_living_sector": living_sector,
            "agent_living_sector_areas": ", ".join(living_sector_arenas) if living_sector_arenas else ""
        }
        arena_prompt = self._render_prompt("action_arena_v2.txt", arena_prompt_context)
        logger.debug(f"[Planner] Arena prompt =>\n{arena_prompt}")
        raw_arena = self.model.generate(
            arena_prompt,
            max_tokens=32,
            stop=["User:", "###"],
            temperature=0.25,
            top_p=0.9,
            allowed_strings=available_arenas,
        ).strip()
        logger.debug(
            "[Planner] Arena raw output='%s', options=%s",
            raw_arena,
            available_arenas,
        )
        default_arena = None
        try:
            current_arena = world.tile_manager.get_tile(agent.position[0], agent.position[1]).arena
        except Exception:
            current_arena = None
        try:
            living_parts = (agent.living_area or "").split(":")
            living_sector_default = living_parts[1] if len(living_parts) >= 2 else ""
            living_arena_default = living_parts[2] if len(living_parts) >= 3 else ""
        except Exception:
            living_sector_default, living_arena_default = "", ""
        if choosen_sector == living_sector_default and living_arena_default in available_arenas:
            default_arena = living_arena_default
        elif current_arena in available_arenas:
            default_arena = current_arena
        elif available_arenas:
            default_arena = available_arenas[0]

        arena = self._pick_from_options(
            raw_arena,
            available_arenas,
            default_arena or (available_arenas[0] if available_arenas else "")
        )

        # Determine Action object
        tm = world.tile_manager
        available_objects = agent.spatial_memory.get_game_objects_in_arena(world.name, choosen_sector, arena)

        def tile_scan_objects() -> List[str]:
            try:
                poss = tm.find_positions(sector=choosen_sector, arena=arena, include_collidable=True)
                seen: List[str] = []
                seen_set = set()
                for (x, y) in poss:
                    g = tm.get_tile(x, y).game_object
                    if g and g not in seen_set:
                        seen.append(g)
                        seen_set.add(g)
                logger.debug("[Planner] Tile-scan objects for sector='%s', arena='%s': %s", choosen_sector, arena, seen)
                return seen
            except Exception as e:
                logger.debug("[Planner] Could not build objects via tile scan: %s", e)
                return []

        available_objects = tile_scan_objects()

        # Compute a sensible default
        default_object = None
        if choosen_sector == current_sector and arena == current_arena:
            try:
                current_object = world.tile_manager.get_tile_path(agent.position, "game_object").split(":")[-1]
                if current_object in available_objects:
                    default_object = current_object
            except Exception:
                pass
        if not default_object:
            default_object = available_objects[0] if available_objects else ""

        logger.debug("[Planner] Object options: %s", available_objects)

        object_prompt_context = {
            "agent": agent.get_state(),
            "arena": arena,
            "available_objects": available_objects,
            "task_description": current_task_desc
        }
        if not available_objects:
            logger.debug("[Planner] No objects available, using default='%s' for task='%s'", default_object or "", current_task_desc)
            game_object = default_object
        else:
            object_prompt = self._render_prompt("action_game_object.txt", object_prompt_context)
            logger.debug(f"[Planner] Object prompt =>\n{object_prompt}")
            raw_object = self.model.generate(
                object_prompt,
                max_tokens=32,
                stop=["User:", "###"],
                temperature=0.25,
                top_p=0.9,
                allowed_strings=available_objects,
            ).strip()
            logger.debug("[Planner] Object raw output='%s', options=%s", raw_object, available_objects)
            game_object = self._pick_from_options(raw_object, available_objects, default_object)

        address = f"{world.name}:{choosen_sector}:{arena}" + (f":{game_object}" if game_object else "")

        duration = min(60, max(5, remaining_minutes))

        try:
            logger.debug(f"[Planner] Generating event triple for action '{current_task_desc}'...")
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
            logger.debug(f"[Planner] Object interaction triples: {obj_triples}")
            if obj_triples:
                s, p, o = obj_triples[0]
                agent.short_term_memory.action.object_interaction.event = Event(s, p, o, obj_desc)
        except Exception as e:
            logger.debug(f"[Planner] Object interaction generation skipped: {e}")

        try:
            end_time = (now + timedelta(minutes=duration)).strftime("%H:%M")
            micro = self.generate_microtasks(
                agent,
                world,
                task_description=current_task_desc,
                start_time=time_key,
                end_time=end_time,
                duration_minutes=duration,
                sector=choosen_sector,
                arena=arena,
            )
            agent.short_term_memory.action.subtasks = self._parse_microtasks_text(micro)
        except Exception as e:
            logger.debug(f"[Planner] Microtasks not generated: {e}")

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



    def _decide_reaction(self, agent: Agent, world: World, context_bundle: Dict[str, List[MemoryNode]]) -> Optional[str]:
        logger.info("[Planner] Deciding whether to react to an event...")
        event = context_bundle["current_event"]
        try:
            max_prompt_mem = int(os.environ.get("MAX_PROMPT_MEMORIES", "20"))
        except Exception:
            max_prompt_mem = 20
        logger.debug(f"[Planner] Context Bundle keys: {list(context_bundle.keys())}")
        logger.debug(f"[Planner] Context Bundle values: {[len(v) if isinstance(v, list) else type(v) for v in context_bundle.values()]}")
        context_nodes = context_bundle.get("context_nodes", [])
        logger.debug(f"[Planner] Context nodes count: {len(context_nodes)}")
        
        context_str = ""
        for n in context_nodes:
            if getattr(n, "node_type", "") == "event" and n.event and n.event.description:
                curr_desc = n.event.description.split(" ")
                if len(curr_desc) > 2:
                    curr_desc[2:3] = ["was"]
                context_str += " ".join(curr_desc) + ". "
            elif getattr(n, "node_type", "") == "thought" and getattr(n, "event", None) and getattr(n.event, "description", None):
                context_str += n.event.description + ". "

        memories = [
            n.event.description for n in context_nodes
            if getattr(n, "event", None) and getattr(n.event, "description", None)
        ][:max(0, max_prompt_mem)]
        logger.debug(f"[Planner] Using {len(memories)} context memories for reaction decision.")
        now = agent.short_term_memory.current_time
        current_time = now.strftime("%B %d, %Y, at %I:%M %p") if now else "an unknown hour"
        current_desc = (agent.short_term_memory.action.description or "").lower()
        if agent.short_term_memory.action.chat.with_whom:
            logger.debug("[Planner] Agent is in a chat; skipping reaction.")
            return None
        
        try:
            current_sector = world.tile_manager.get_current_sector(agent.position)
        except Exception:
            current_sector = ""
        try:
            current_arena = world.tile_manager.get_tile_path(agent.position, "arena").split(":")[-1]
        except Exception:
            current_arena = ""
        try:
            current_object = world.tile_manager.get_tile_path(agent.position, "game_object").split(":")[-1]
        except Exception:
            current_object = ""
        addr_parts: List[str] = [world.name]
        if current_sector:
            addr_parts.append(current_sector)
        if current_arena:
            addr_parts.append(current_arena)
        if current_object:
            addr_parts.append(current_object)
        current_address = ":".join(addr_parts)
        is_at_location = current_address == agent.short_term_memory.action.address
        agent_loc = ""
        if agent.short_term_memory.action.address and ':' in agent.short_term_memory.action.address:
            parts = agent.short_term_memory.action.address.split(':')
            agent_loc = parts[-1] + " in " + parts[-2] if len(parts) >= 2 else ""
        init_p_desc = f"{agent.name} is already {current_desc} at {agent_loc}" if is_at_location else f"{agent.name} is on the way to {current_desc} at {agent_loc}"

        target_name = event.subject
        target_act_desc = f"{event.predicate} {event.object}" if event.object else event.predicate
        target_p_desc = f"{target_name} {target_act_desc}"
        target_loc = ""
        target_agent = None
        try:
            target_agent = next((a for a in world.agents if a.name == target_name), None)
            if target_agent:
                target_act_desc = target_agent.short_term_memory.action.description or "idle"
                target_address = target_agent.short_term_memory.action.address or ""
                if ':' in target_address:
                    parts = target_address.split(':')
                    target_loc = parts[-1] + " in " + parts[-2] if len(parts) >= 2 else ""
                target_current_sector = world.tile_manager.get_current_sector(target_agent.position)
                target_current_arena = world.tile_manager.get_tile_path(target_agent.position, "arena").split(":")[-1]
                target_current_object = world.tile_manager.get_tile_path(target_agent.position, "game_object").split(":")[-1]
                target_current_address = f"{world.name}:{target_current_sector}:{target_current_arena}" + (f":{target_current_object}" if target_current_object else "")
                target_is_at = target_current_address == target_address
                target_p_desc = f"{target_name} is already {target_act_desc} at {target_loc}" if target_is_at else f"{target_name} is on the way to {target_act_desc} at {target_loc}"
        except Exception as e:
            logger.debug(f"[Planner] Could not retrieve target agent details: {e}")

        move_targets: List[str] = []
        world_name = world.name
        sectors = agent.spatial_memory.get_accessible_sectors(world_name) or []
        if not sectors:
            try:
                tm = world.tile_manager
                seen: set[str] = set()
                for x in range(tm.width):
                    for y in range(tm.height):
                        t = tm.get_tile(x, y)
                        if t.sector:
                            seen.add(t.sector)
                sectors = sorted(seen)
            except Exception:
                sectors = []
        try:
            tm = world.tile_manager
        except Exception:
            tm = None
        dedup: set[str] = set()
        for s in sectors:
            addr2 = f"{world_name}:{s}"
            if addr2 not in dedup:
                move_targets.append(addr2)
                dedup.add(addr2)
            arenas = []
            if tm is not None:
                try:
                    poss = tm.find_positions(sector=s, include_collidable=True)
                    seen_arenas: set[str] = set()
                    for (x, y) in poss:
                        a = tm.get_tile(x, y).arena
                        if a and a not in seen_arenas:
                            seen_arenas.add(a)
                    arenas = sorted(seen_arenas)
                except Exception:
                    arenas = []
            for a in arenas:
                addr3 = f"{world_name}:{s}:{a}"
                if addr3 not in dedup:
                    move_targets.append(addr3)
                    dedup.add(addr3)
        
        context = {
            "agent": agent.get_state(),
            "event": {
                "subject": event.subject,
                "predicate": event.predicate or "is",
                "object": event.object or "idle",
                "description": event.description,
            },
            "memories": memories,
            "current_time": current_time,
            "current_action": current_desc,
            "in_chat": bool(agent.short_term_memory.action.chat.with_whom),
            "move_targets": move_targets,
            "context": context_str,
            "init_p_desc": init_p_desc,
            "target_p_desc": target_p_desc,
            "target_act_desc": target_act_desc,
        }
        prompt = self._render_prompt("should_react_to_event.txt", context)
        logger.debug(f"[Planner] Reaction decision prompt =>\n{prompt}")
        
        response = self.model.generate(
            prompt,
            max_tokens=200,
            stop=[":"],
            temperature=0,
            top_p=1,
        )

        option = ""
        try:
            option_part = response.split("Option ")[-1].strip().lower()
            option = option_part[0]
        except Exception:
            option = "2"
        
        logger.debug(f"[Planner] Raw reaction response: {response}")
        
        if option == "1":
            return "chat with " + (event.subject or "someone")
        elif option == "2":
            return "ignore"
        else:
            return None

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
            target_text = reaction_mode[len("chat with"):].strip()
            target = target_text or event.subject or event.object or "someone"
            try:
                if now and now.hour == 23:
                    logger.debug("[Planner] Skipping chat: quiet hours.")
                    return
                if agent.short_term_memory.action.chat.with_whom:
                    logger.debug("[Planner] Skipping chat: initiator already in chat.")
                    return
                target_agent = next((a for a in world.agents if a.name == target), None)
                if target_agent is None:
                    logger.debug("[Planner] Skipping chat: target not found.")
                    return
                if target_agent.short_term_memory.action.chat.with_whom:
                    logger.debug("[Planner] Skipping chat: target already in chat.")
                    return
                if not (agent.short_term_memory.action.description or "").strip() or not (target_agent.short_term_memory.action.description or "").strip():
                    logger.debug("[Planner] Skipping chat: missing action descriptions.")
                    return
                if "sleep" in (agent.short_term_memory.action.description or "").lower() or "sleep" in (target_agent.short_term_memory.action.description or "").lower():
                    logger.debug("[Planner] Skipping chat: sleep in progress.")
                    return
                if "waiting" in (target_agent.short_term_memory.action.description or "").lower():
                    logger.debug("[Planner] Skipping chat: target waiting.")
                    return
                init_buf = agent.short_term_memory.action.chat.buffer or {}
                targ_buf = target_agent.short_term_memory.action.chat.buffer or {}
                if init_buf.get(target, 0) > 0:
                    logger.debug("[Planner] Skipping chat: initiator cooldown active for %s (%d)", target, init_buf.get(target, 0))
                    return
                if targ_buf.get(agent.name, 0) > 0:
                    logger.debug("[Planner] Skipping chat: target cooldown active for %s (%d)", agent.name, targ_buf.get(agent.name, 0))
                    return
            except Exception as e:
                logger.debug("[Planner] Chat guard checks failed: %s", e)
                return

            convo = self.chat_manager.generate_conversation(agent, world, target)
            summary = self.chat_manager.summarize_conversation(agent, convo)
            turns = max(1, len(convo))
            minutes = max(5, min(20, (turns + 1) // 2))

            agent.short_term_memory.action.chat.with_whom = target
            agent.short_term_memory.action.chat.chat_log = [[s, u] for s, u in convo]

            meeting_addr = None
            try:
                tm = world.tile_manager
                initiator_pos = agent.position
                if target_agent is not None:
                    tx, ty = target_agent.position
                    occupied = {a.position for a in world.agents}
                    candidates_adj = []
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = tx + dx, ty + dy
                        if not tm.is_within_bounds(nx, ny):
                            continue
                        if tm.is_collidable(nx, ny):
                            continue
                        if (nx, ny) in occupied:
                            continue
                        candidates_adj.append((nx, ny))
                    if candidates_adj:
                        mx, my = min(candidates_adj, key=lambda p: (p[0] - initiator_pos[0]) ** 2 + (p[1] - initiator_pos[1]) ** 2)
                    else:
                        mx, my = tx, ty
                    meeting_addr = f"<waiting> {mx} {my}"
                    agent.short_term_memory.action.address = meeting_addr
            except Exception:
                meeting_addr = None

            agent.short_term_memory.action.description = f"heading to chat with {target}"
            agent.short_term_memory.action.event = Event(agent.name, "chat with", target, summary)
            agent.short_term_memory.action.duration = minutes
            agent.short_term_memory.action.start(now)
            logger.debug(f"[Planner] Planned action to chat with {target} for {minutes} minutes. Summary: {summary}")

            try:
                target_agent: Agent = next((a for a in world.agents if a.name == target), None)
                if target_agent is not None:
                    target_agent.short_term_memory.action.chat.with_whom = agent.name
                    target_agent.short_term_memory.action.chat.chat_log = [[s, u] for s, u in convo]
                    target_agent.short_term_memory.action.description = f"heading to chat with {agent.name}"
                    target_agent.short_term_memory.action.event = Event(target_agent.name, "chat with", agent.name, summary)
                    target_agent.short_term_memory.action.duration = minutes
                    target_agent.short_term_memory.action.start(now)
                    if meeting_addr:
                        target_agent.short_term_memory.action.address = meeting_addr
                        logger.debug(f"[Planner] Meeting address for chat set to {meeting_addr}.")
                    logger.debug(f"[Planner] Notified {target_agent.name} action to chat with {agent.name}.")
            except Exception:
                pass

        elif reaction_mode.startswith("wait"):
            logger.debug(f"[Planner] Setting up wait reaction to event: {event.subject} {event.predicate} {event.object}")
            agent.short_term_memory.action.description = "waiting quietly"
            agent.short_term_memory.action.event = Event(agent.name, "is", "waiting", "The agent decides to wait quietly.")
            agent.short_term_memory.action.duration = 10
            agent.short_term_memory.action.start(now)
            logger.debug(f"[Planner] Planned action to wait for 10 minutes.")
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
        logger.debug(f"[Planner] Object interaction response: {response}")
        return response

    def _parse_microtasks_text(self, text: str) -> List[str]:
        raw_lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

        import re
        time_marker = re.compile(r"(?=(?:\b\d{1,2}:\d{2}\b\s*:)|(?:\b\d{1,2}\b\s*:))")
        lines: List[str] = []
        for l in raw_lines:
            parts = [p.strip() for p in time_marker.split(l) if p and not time_marker.fullmatch(p)]
            if parts:
                acc: List[str] = []
                curr = ""
                for chunk in parts:
                    if re.match(r"^(?:\d{1,2}:\d{2}|\d{1,2})\s*:\s*", chunk):
                        if curr:
                            acc.append(curr.strip())
                        curr = chunk
                    else:
                        curr = (curr + " " + chunk).strip()
                if curr:
                    acc.append(curr.strip())
                lines.extend(acc)
            else:
                lines.append(l)

        cleaned: List[str] = []
        for l in lines:
            s = l
            # Drop bullets/numbers
            s = re.sub(r"^[\u2022•\-\*\d\)\.\s]+", "", s)
            # Remove time ranges like "HH:MM - HH:MM -" or single "HH:MM:"
            s = re.sub(r"^(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:-\s*\d{1,2}:\d{2}(?::\d{2})?)?\s*[-:]?\s*)+", "", s)
            # Remove stray leading ":MM:" leftovers
            s = re.sub(r"^:\d{1,2}:\s*", "", s)
            # Remove bare minute markers like "05:" at start
            s = re.sub(r"^\b\d{1,2}\b\s*:\s*", "", s)
            # Normalize whitespace
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                cleaned.append(s)
        return cleaned
