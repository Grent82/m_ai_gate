from typing import List, Tuple

from core.agent_action import Event
from core.world import World

class WorldBuilder:
    def __init__(self):
        pass

    def setup_medieval_village_world(self) -> World:
        world = World(name="MedievalVillage", description="A small medieval hamlet with two houses, a tavern, a field and a forest", width=20, height=20)
        tiles = world.tile_manager
        for x in range(world.tile_manager.width):
            for y in range(world.tile_manager.height):
                tiles.set_sector(x, y, "Outside")
                tiles.set_arena(x, y, "Village Outskirts")
                tiles.set_game_object(x, y, "grass")
                tiles.set_collision(x, y, False)

        def build_room(
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            *,
            sector: str,
            arena: str,
            objects: List[Tuple[int, int, str]] = None,
            doors: List[Tuple[int, int]] = None,
        ) -> None:
            """Build a rectangular room with walls, floor, and optional doors.

            - Perimeter becomes `wall` (collidable) except at `doors` which become
              `door` (non-collidable)
            - Interior becomes `floor` (non-collidable)
            - Places any `objects` provided on top (does not change collision)
            - Validates bounds and coordinate ordering
            """
            if x1 > x2 or y1 > y2:
                raise ValueError("Invalid room bounds: x1<=x2 and y1<=y2 are required.")
            if not (0 <= x1 < tiles.width and 0 <= x2 < tiles.width and 0 <= y1 < tiles.height and 0 <= y2 < tiles.height):
                raise ValueError("Room coordinates out of world bounds.")

            doors = doors or []
            door_set = set(doors)
            objects = objects or []

            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    on_perimeter = x in (x1, x2) or y in (y1, y2)
                    tiles.set_sector(x, y, sector)
                    tiles.set_arena(x, y, arena)

                    if on_perimeter:
                        if (x, y) in door_set:
                            tiles.set_game_object(x, y, "door")
                            tiles.set_collision(x, y, False)
                        else:
                            tiles.set_game_object(x, y, "wall")
                            tiles.set_collision(x, y, True)
                    else:
                        tiles.set_game_object(x, y, "floor")
                        tiles.set_collision(x, y, False)

            # Object collision semantics: items you use (bed, bench, chair) remain passable;
            # bulky fixtures block movement.
            passable_furniture = {"bed", "bench", "chair"}
            blocking_objects = {
                "table",
                "bar counter",
                "fireplace",
                "wooden chest",
                "workbench",
                "crate",
                "beer barrel",
            }
            for (ox, oy, obj_name) in objects:
                tiles.set_game_object(ox, oy, obj_name)
                if obj_name in passable_furniture:
                    tiles.set_collision(ox, oy, False)
                elif obj_name in blocking_objects:
                    tiles.set_collision(ox, oy, True)
                # else: leave existing collision as is

        # House 1 – Farmer
        build_room(
            1, 1, 5, 5,
            sector="Farmer's House",
            arena="Main Room",
            objects=[
                (2, 2, "bed"),
                (3, 2, "fireplace"),
                (2, 3, "table"),
                (3, 3, "crate")
            ],
            doors=[(3, 1)]  # north door
        )

        # House 2 – Hunter
        build_room(
            8, 1, 12, 5,
            sector="Hunter's Cabin",
            arena="Main Room",
            objects=[
                (9, 2, "bed"),
                (10, 2, "fireplace"),
                (9, 3, "wooden chest"),
                (10, 3, "workbench")
            ],
            doors=[(10, 1)]  # north door
        )

        # Tavern
        build_room(
            4, 8, 10, 13,
            sector="The Drunken Boar Tavern",
            arena="Tavern Hall",
            objects=[
                (5, 9, "bar counter"),
                (5, 10, "beer barrel"),
                (6, 11, "table"),
                (7, 11, "bench"),
                (8, 11, "bench"),
                (6, 12, "fireplace"),
                (9, 12, "bed")  # bed for innkeeper
            ],
            doors=[(7, 8)]  # north door
        )

        # Add field area (next to farmer's house)
        for x in range(1, 6):
            # Limit to y=6..7 to avoid overlapping the tavern's top wall at y=8
            for y in range(6, 8):
                tiles.set_sector(x, y, "Farmer's Field")
                tiles.set_arena(x, y, "Wheat Patch")
                tiles.set_game_object(x, y, "wheat" if (x + y) % 2 == 0 else "soil")
                tiles.set_collision(x, y, False)

        # Add forest area (near hunter’s cabin)
        for x in range(13, 18):
            for y in range(1, 6):
                tiles.set_sector(x, y, "Forest Edge")
                tiles.set_arena(x, y, "Hunting Grounds")
                obj = "tree" if (x + y) % 2 == 0 else "bush"
                tiles.set_game_object(x, y, obj)
                # Trees are collidable, bushes are passable
                tiles.set_collision(x, y, obj == "tree")

        # Add simple gravel paths to make movement easier/preferred
        def lay_gravel_path(points: List[Tuple[int, int]]):
            for (px, py) in points:
                if not (0 <= px < tiles.width and 0 <= py < tiles.height):
                    continue
                # Do not overwrite doors or walls
                obj_here = tiles.get_tile(px, py).game_object
                if obj_here in {"door", "wall"}:
                    continue
                tiles.set_arena(px, py, "Village Path")
                tiles.set_game_object(px, py, "gravel")
                tiles.set_collision(px, py, False)

        # Connect both house entrances to tavern via a T-shaped path
        gravel_points = set()
        # Top horizontal from farmer (x=3) to hunter (x=10) along y=0, centered at x=7
        for x in range(3, 11):
            gravel_points.add((x, 0))
        # Vertical down from x=7 (center) to just before the tavern door at (7,8)
        for y in range(0, 8):  # stop at y=7 to avoid overwriting the door at y=8
            gravel_points.add((7, y))
        lay_gravel_path(sorted(gravel_points))

        # Tavern event
        tiles.add_event_to_tile(
            6, 11,
            Event(
                subject="Innkeeper",
                predicate="talks to",
                object="traveler",
                description="The innkeeper is welcoming a traveler with ale."
            )
        )

        return world
