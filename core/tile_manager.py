from typing import List, Tuple, Optional

from core.agent_action import Event
from core.tile import Tile
from core.logger import setup_logger

logger = setup_logger(__name__)

class TileManager:
    """
    Manages a grid of tiles within a world, allowing access to tile properties and operations.
    """

    def __init__(self, width: int, height: int, world_name: str):
        """
        Initializes the TileManager.

        :param width: Width of the tile grid.
        :param height: Height of the tile grid.
        :param world_name: Name of the world the tiles belong to.
        """
        self.width = width
        self.height = height
        self.world_name = world_name
        self.tiles = [
            [Tile(world_name, None, None, None, None) for _ in range(height)]
            for _ in range(width)
        ]
        # todo, obstacles

    def get_tile(self, x: int, y: int) -> Tile:
        """
        Retrieves the tile at the given coordinates.

        :param x: X-coordinate of the tile.
        :param y: Y-coordinate of the tile.
        :return: Tile instance.
        :raises ValueError: If coordinates are out of bounds.
        """
        if not self.is_within_bounds(x, y):
            logger.error("Coordinates (%d, %d) are out of bounds.", x, y)
            raise ValueError("Coordinates are out of bounds.")
        return self.tiles[x][y]

    def get_current_sector(self, position: Tuple[int, int]) -> str:
        """
        Retrieves the sector of the tile at the given position.

        :param position: (x, y) tuple of the tile position.
        :return: Sector name.
        :raises ValueError: If the sector is undefined.
        """
        x, y = position
        tile = self.get_tile(x, y)
        if not tile.sector:
            logger.error("Sector not defined for tile at position (%d, %d).", x, y)
            raise ValueError("Sector not defined for the tile.")
        return tile.sector

    def get_nearby_tiles_positions(
        self, position: Tuple[int, int], radius: int
    ) -> List[Tuple[int, int]]:
        """
        Retrieves positions of tiles within a given radius of a position.

        :param position: Center position as (x, y).
        :param radius: Radius to search for nearby tiles.
        :return: List of (x, y) tuples for nearby tiles.
        """
        x, y = position
        return [
            (i, j)
            for i in range(max(0, x - radius), min(self.width, x + radius + 1))
            for j in range(max(0, y - radius), min(self.height, y + radius + 1))
        ]

    def get_tile_path(self, position: Tuple[int, int], level: str) -> str:
        """
        Constructs a hierarchical path for a tile up to a specific level.

        :param position: (x, y) tuple of the tile position.
        :param level: Path level ("world", "sector", "arena", "game_object").
        :return: Path as a colon-separated string.
        :raises ValueError: If the level is invalid or data is missing.
        """
        x, y = position
        tile = self.get_tile(x, y)
        path = [self.world_name, tile.sector, tile.arena, tile.game_object]
        levels = ["world", "sector", "arena", "game_object"]
        if level not in levels:
            logger.error("Invalid level: %s", level)
            raise ValueError("Invalid level.")
        index = levels.index(level) + 1
        if any(p is None for p in path[:index]):
            logger.error(
                "Path level %s is missing for position (%d, %d).", level, x, y
            )
            raise ValueError("Path level is missing.")
        return ":".join(filter(None, path[:index]))

    def set_sector(self, x: int, y: int, sector: str) -> None:
        """Sets the sector for a specific tile."""
        self.get_tile(x, y).sector = sector

    def set_arena(self, x: int, y: int, arena: str) -> None:
        """Sets the arena for a specific tile."""
        self.get_tile(x, y).arena = arena

    def set_game_object(self, x: int, y: int, game_object: str) -> None:
        """Sets the game object for a specific tile."""
        self.get_tile(x, y).game_object = game_object

    def set_collision(self, x: int, y: int, collision: bool) -> None:
        """Sets the collision status for a specific tile."""
        self.get_tile(x, y).collision = collision

    def is_collidable(self, x: int, y: int) -> bool:
        """Checks if a tile is collidable."""
        return self.get_tile(x, y).is_collidable()

    def add_event_to_tile(self, x: int, y: int, event: Event) -> None:
        """Adds an event to a tile."""
        self.get_tile(x, y).add_event(event)

    def remove_event_from_tile(self, x: int, y: int, event: Event) -> None:
        """Removes an event from a tile."""
        self.get_tile(x, y).remove_event(event)

    def is_within_bounds(self, x: int, y: int) -> bool:
        """
        Checks if coordinates are within the bounds of the grid.

        :param x: X-coordinate.
        :param y: Y-coordinate.
        :return: True if within bounds, False otherwise.
        """
        return 0 <= x < self.width and 0 <= y < self.height

    # --- Added helpers for execution/path planning ---
    def find_positions(
        self,
        sector: Optional[str] = None,
        arena: Optional[str] = None,
        game_object: Optional[str] = None,
        include_collidable: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Find all tile positions that match the requested attributes.

        Any of sector/arena/game_object may be None to act as a wildcard.
        """
        matches: List[Tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                t = self.tiles[x][y]
                if sector is not None and t.sector != sector:
                    continue
                if arena is not None and t.arena != arena:
                    continue
                if game_object is not None and t.game_object != game_object:
                    continue
                if not include_collidable and t.is_collidable():
                    continue
                matches.append((x, y))
        return matches

    def find_positions_by_address(self, address: str) -> List[Tuple[int, int]]:
        """
        Parse an address in the form 'world:sector:arena:object' and find positions.

        The world name is ignored (this manager already knows its world).
        Missing parts are treated as wildcards.
        """
        parts = [p.strip() for p in address.split(":")]
        # Expected formats:
        # [world]
        # [world, sector]
        # [world, sector, arena]
        # [world, sector, arena, object]
        sector = None
        arena = None
        obj = None
        if len(parts) >= 2:
            sector = parts[1] or None
        if len(parts) >= 3:
            arena = parts[2] or None
        if len(parts) >= 4:
            obj = parts[3] or None
        return self.find_positions(sector=sector, arena=arena, game_object=obj)
