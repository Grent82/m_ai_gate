from typing import List, Tuple

from core.agent_action import Event
from core.world import World
from core.tile import TerrainType

class WorldBuilder:
    def __init__(self):
        pass

    def setup_medieval_village_world(self) -> World:
        world = World(name="MedievalVillage", description="A small medieval hamlet with two houses, a tavern, a field and a forest", width=30, height=24)
        tiles = world.tile_manager
        for x in range(world.tile_manager.width):
            for y in range(world.tile_manager.height):
                tiles.set_sector(x, y, "Outside")
                tiles.set_arena(x, y, "Village Outskirts")
                tiles.set_terrain(x, y, TerrainType.GRASS)
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

            - Perimeter is collidable (walls); door coordinates are non-collidable openings
            - Interior becomes floor (non-collidable)
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
                            tiles.set_collision(x, y, False)
                        else:
                            tiles.set_terrain(x, y, TerrainType.WALL)
                            tiles.set_collision(x, y, True)
                    else:
                        tiles.set_terrain(x, y, TerrainType.FLOOR)
                        tiles.set_collision(x, y, False)

            passable_furniture = {"bed", "bench", "chair"}
            blocking_objects = {
                "table",
                "bar counter",
                "fireplace",
                "wooden chest",
                "workbench",
                "crate",
                "beer barrel",
                # smithery equipment
                "anvil",
                "forge",
                "tool rack",
                "grindstone",
            }
            for (ox, oy, obj_name) in objects:
                tiles.set_game_object(ox, oy, obj_name)
                if obj_name in passable_furniture:
                    tiles.set_collision(ox, oy, False)
                elif obj_name in blocking_objects:
                    tiles.set_collision(ox, oy, True)

        # House 1 – Farmer
        build_room(
            1, 1, 6, 6,
            sector="Farmer's House",
            arena="Private Room",
            objects=[
                (2, 2, "bed"),
                (5, 2, "fireplace"),
                (2, 3, "table"),
                (5, 5, "crate"),
                (3, 3, "chair")
            ],
            doors=[(3, 1)]  # north door
        )

        # House 3 – Blacksmith (Forge + Private Room)
        # Smithery workspace
        build_room(
            18, 10, 25, 15,
            sector="Blacksmith's Forge",
            arena="Smithery",
            objects=[
                (20, 12, "anvil"),
                (22, 12, "forge"),
                (19, 13, "tool rack"),
                (23, 13, "workbench"),
                (24, 14, "water bucket"),
                (21, 14, "grindstone"),
            ],
            # West-facing front doors to the road and an internal door to private room
            doors=[(18, 11), (18, 12), (20, 15)]  # (20,15) connects to private room below
        )

        # Private room for the blacksmith and spouse (attached south of smithery)
        # Enlarged to provide clear walking space
        build_room(
            17, 16, 24, 21,
            sector="Blacksmith's Forge",
            arena="Private Room",
            objects=[
                (18, 17, "bed"),
                (23, 17, "wooden chest"),
                (21, 19, "table"),
                (22, 19, "chair"),
                (18, 20, "fireplace"),
                (19, 20, "bench"),
            ],
            doors=[(20, 16)]  # north door aligned to (20,15) in smithery
        )

        # House 2 – Hunter
        build_room(
            8, 1, 14, 6,
            sector="Hunter's Cabin",
            arena="Private Room",
            objects=[
                (9, 2, "bed"),
                (13, 2, "fireplace"),
                (9, 3, "wooden chest"),
                (11, 3, "workbench"),
                (12, 4, "table"),
                (12, 5, "chair")
            ],
            doors=[(10, 1)]  # north door
        )

        # Tavern Hall (main room)
        build_room(
            4, 8, 12, 16,
            sector="The Drunken Boar Tavern",
            arena="Tavern Hall",
            objects=[
                (5, 9, "bar counter"),
                (5, 10, "beer barrel"),
                (6, 11, "table"),
                (7, 11, "bench"),
                (8, 11, "bench"),
                (6, 12, "fireplace"),
                # Additional seating
                (6, 10, "chair"),
                (6, 13, "chair"),
                (10, 14, "table"),
                (9, 14, "chair"),
                (11, 14, "chair")
            ],
            # front door and internal door on the east wall to the private room (12,14)
            doors=[(7, 8), (12, 14)]  # north door and side door to private room
        )

        # Tavern Private Room (for the innkeeper), attached east of the hall
        build_room(
            13, 12, 17, 16,
            sector="The Drunken Boar Tavern",
            arena="Private Room",
            objects=[
                (15, 14, "bed"),
                (16, 14, "wooden chest")
            ],
            # Door aligned with hall opening at (12,14)
            doors=[(13, 14)]
        )

        # Add field area (next to farmer's house)
        for x in range(1, 7):
            for y in range(7, 8):
                tiles.set_sector(x, y, "Farmer's Field")
                tiles.set_arena(x, y, "Wheat Patch")
                tiles.set_terrain(x, y, TerrainType.SOIL)
                if (x + y) % 2 == 0:
                    tiles.set_game_object(x, y, "wheat")
                else:
                    tiles.set_game_object(x, y, "") # TODO: let the wheat grow back over time
                tiles.set_collision(x, y, False)

        # Add forest area (near hunter’s cabin)
        for x in range(16, 26):
            for y in range(1, 9):
                tiles.set_sector(x, y, "Forest Edge")
                tiles.set_arena(x, y, "Hunting Grounds")
                tiles.set_terrain(x, y, TerrainType.FOREST)
                # Keep some density but avoid isolating passable tiles (no checkerboard walls)
                is_dense = (x % 3 == 0 and y % 2 == 0)
                tiles.set_collision(x, y, is_dense)

        # Ensure a walkable corridor from the village road into the forest
        for y in range(1, 9):
            for x in (17, 18):
                if 16 <= x < 26:
                    tiles.set_collision(x, y, False)

        def lay_gravel_path(points: List[Tuple[int, int]]):
            for (px, py) in points:
                if not (0 <= px < tiles.width and 0 <= py < tiles.height):
                    continue
                if tiles.get_tile(px, py).is_collidable():
                    continue
                #tiles.set_arena(px, py, "Village Path")
                tiles.set_terrain(px, py, TerrainType.GRAVEL)
                tiles.set_collision(px, py, False)

        gravel_points = set()
        for x in range(3, 11):
            gravel_points.add((x, 0))
        for y in range(0, 8):
            gravel_points.add((7, y))
        for x in range(2, 28):
            gravel_points.add((x, 2))
        for y in range(2, 20):
            gravel_points.add((15, y))
        lay_gravel_path(sorted(gravel_points))

        # Connect main north-south road (x=15) to the blacksmith entrances at (18,11) and (18,12)
        lay_gravel_path([(16, 11), (17, 11), (18, 11)])
        lay_gravel_path([(16, 12), (17, 12), (18, 12)])

        tiles.set_game_object(1, 7, "water bucket")
        tiles.set_collision(1, 7, False)
        tiles.set_game_object(0, 3, "wood pile")
        tiles.set_collision(0, 3, True)

        # Near Hunter's Cabin (forest edge)
        tiles.set_game_object(16, 7, "wood pile")
        tiles.set_collision(16, 7, True)

        # Outside the Tavern (south side)
        tiles.set_game_object(5, 17, "rain barrel")
        tiles.set_collision(5, 17, True)

        return world
