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

        def build_room(x1, y1, x2, y2, sector, arena, objects):
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    if x in (x1, x2) or y in (y1, y2):
                        tiles.set_collision(x, y, True)
                    else:
                        tiles.set_collision(x, y, False)
                    tiles.set_sector(x, y, sector)
                    tiles.set_arena(x, y, arena)
                    tiles.set_game_object(x, y, "floor")
            for (ox, oy, obj_name) in objects:
                tiles.set_game_object(ox, oy, obj_name)

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
            ]
        )

        # House 2 – Hunter
        build_room(
            8, 1, 12, 5,
            sector="Hunter's Cabin",
            arena="Main Room",
            objects=[
                (9, 2, "bed"),
                (10, 2, "hearth"),
                (9, 3, "wooden chest"),
                (10, 3, "workbench")
            ]
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
            ]
        )

        # Add field area (next to farmer's house)
        for x in range(1, 6):
            for y in range(6, 9):
                tiles.set_sector(x, y, "Farmer's Field")
                tiles.set_arena(x, y, "Wheat Patch")
                tiles.set_game_object(x, y, "wheat" if (x + y) % 2 == 0 else "soil")
                tiles.set_collision(x, y, False)

        # Add forest area (near hunter’s cabin)
        for x in range(13, 18):
            for y in range(1, 6):
                tiles.set_sector(x, y, "Forest Edge")
                tiles.set_arena(x, y, "Hunting Grounds")
                tiles.set_game_object(x, y, "tree" if (x + y) % 2 == 0 else "bush")
                tiles.set_collision(x, y, False)

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

