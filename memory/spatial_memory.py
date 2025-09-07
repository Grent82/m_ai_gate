import logging
from typing import Dict, List

from core.logger import setup_logger

logger = setup_logger(__name__)


class SpatialMemory:

    def __init__(self):
        """
        Initializes the spatial memory with an empty spatial tree.
        """
        self.spatial_tree: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    def get_accessible_sectors(self, world_name: str) -> List[str]:
        try:
            return list(self.spatial_tree.get(world_name, {}).keys())
        except Exception as e:
            logger.error("Error retrieving accessible sectors: %s", e)
            return []

    def get_accessible_arenas(self, world_name: str, sector_name: str) -> List[str]:
        try:
            return list(
                self.spatial_tree.get(world_name, {}).get(sector_name, {}).keys()
            )
        except Exception as e:
            logger.error(
                "Error retrieving accessible arenas for world '%s' and sector '%s': %s",
                world_name,
                sector_name,
                e,
            )
            return []

    def update_memory(
        self, world_name: str, sector_name: str, arenas: List[str]
    ) -> None:
        try:
            if not isinstance(arenas, list) or not all(
                isinstance(a, str) for a in arenas
            ):
                logger.error("Invalid arena list format. Expected List[str].")
                return
            self.spatial_tree.setdefault(world_name, {})[sector_name] = arenas
            logger.debug(
                "Updated spatial memory: world='%s', sector='%s', arenas='%s'",
                world_name,
                sector_name,
                arenas,
            )
        except Exception as e:
            logger.error("Error updating spatial memory: %s", e)

    def is_arena_accessible(
        self, world_name: str, sector_name: str, arena_name: str
    ) -> bool:
        try:
            return arena_name in self.get_accessible_arenas(world_name, sector_name)
        except Exception as e:
            logger.error(
                "Error checking arena accessibility for world='%s', sector='%s', arena='%s': %s",
                world_name,
                sector_name,
                arena_name,
                e,
            )
            return False

    def clear_memory(self) -> None:
        try:
            self.spatial_tree.clear()
            logger.debug("Spatial memory cleared.")
        except Exception as e:
            logger.error("Error clearing spatial memory: %s", e)

    def get_game_objects_in_arena(
        self, world_name: str, sector_name: str, arena_name: str
    ) -> List[str]:
        try:
            return (
                self.spatial_tree.get(world_name, {})
                .get(sector_name, {})
                .get(arena_name, [])
            )
        except Exception as e:
            logger.error(
                "Error retrieving game objects for world='%s', sector='%s', arena='%s': %s",
                world_name,
                sector_name,
                arena_name,
                e,
            )
            return []

    def update_arena_objects(
        self, world_name: str, sector_name: str, arena_name: str, objects: List[str]
    ) -> None:
        try:
            if not isinstance(objects, list) or not all(isinstance(o, str) for o in objects):
                logger.error("Invalid objects format. Expected List[str].")
                return

            self.spatial_tree.setdefault(world_name, {}).setdefault(sector_name, {})[arena_name] = objects

            logger.debug(
                "Updated arena objects: world='%s', sector='%s', arena='%s', objects='%s'",
                world_name,
                sector_name,
                arena_name,
                ", ".join(objects),
            )
        except Exception as e:
            logger.error(
                "Error updating arena objects for world='%s', sector='%s', arena='%s': %s",
                world_name,
                sector_name,
                arena_name,
                e,
            )

