import logging
from typing import Dict, List

from core.logger import setup_logger

logger = setup_logger(__name__)


class SpatialMemory:

    def __init__(self):
        """
        Initializes the spatial memory with an empty spatial tree.
        """
        # Structure: world -> sector -> arena -> [objects]
        self.spatial_tree: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    def get_accessible_sectors(self, world_name: str) -> List[str]:
        try:
            sectors = self.spatial_tree.get(world_name, {})
            if isinstance(sectors, dict):
                result = list(sectors.keys())
                logger.debug("Accessible sectors for world '%s': %s", world_name, result)
                return result
            # Repair if legacy list slipped in
            logger.warning("Unexpected sectors type for world '%s': %s", world_name, type(sectors).__name__)
            logger.debug("World '%s' sectors raw value: %r", world_name, sectors)
            return []
        except Exception as e:
            logger.error("Error retrieving accessible sectors: %s", e)
            return []

    def get_accessible_arenas(self, world_name: str, sector_name: str) -> List[str]:
        try:
            sector_map = self.spatial_tree.get(world_name, {}).get(sector_name, {})
            if isinstance(sector_map, dict):
                result = list(sector_map.keys())
                logger.debug(
                    "Accessible arenas for world '%s', sector '%s': %s",
                    world_name,
                    sector_name,
                    result,
                )
                return result
            logger.warning(
                "Unexpected arenas type for world '%s', sector '%s': %s",
                world_name,
                sector_name,
                type(sector_map).__name__,
            )
            logger.debug(
                "World '%s', sector '%s' arenas raw value: %r",
                world_name,
                sector_name,
                sector_map,
            )
            return []
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
        """
        Ensure the given arenas exist under the world/sector, initializing each arena with an empty
        object list if not present. Does not overwrite existing arena object lists.
        """
        try:
            if not isinstance(arenas, list) or not all(isinstance(a, str) for a in arenas):
                logger.error("Invalid arena list format. Expected List[str].")
                return

            world_map = self.spatial_tree.setdefault(world_name, {})
            sector_map = world_map.setdefault(sector_name, {})
            if not isinstance(sector_map, dict):
                # Repair legacy bad state
                logger.warning(
                    "Repairing sector map for world='%s', sector='%s' from type %s to dict.",
                    world_name,
                    sector_name,
                    type(sector_map).__name__,
                )
                sector_map = {}
                world_map[sector_name] = sector_map

            for arena in arenas:
                sector_map.setdefault(arena, [])

            logger.debug(
                "Updated spatial memory: world='%s', sector='%s', arenas='%s'",
                world_name,
                sector_name,
                ", ".join(arenas),
            )
            logger.debug(
                "Current arena keys for world='%s', sector='%s': %s",
                world_name,
                sector_name,
                list(sector_map.keys()),
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
            arena_map = (
                self.spatial_tree.get(world_name, {})
                .get(sector_name, {})
            )
            if not isinstance(arena_map, dict):
                logger.warning(
                    "Unexpected arena map type for world='%s', sector='%s': %s",
                    world_name,
                    sector_name,
                    type(arena_map).__name__,
                )
                logger.debug(
                    "World '%s', sector '%s' arena map raw value: %r",
                    world_name,
                    sector_name,
                    arena_map,
                )
                return []
            objs = arena_map.get(arena_name, [])
            clean = list(dict.fromkeys([o for o in objs if isinstance(o, str) and o]))
            logger.debug(
                "Accessible objects for world='%s', sector='%s', arena='%s': %s",
                world_name,
                sector_name,
                arena_name,
                clean,
            )
            return clean
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

            world_map = self.spatial_tree.setdefault(world_name, {})
            sector_map = world_map.setdefault(sector_name, {})
            if not isinstance(sector_map, dict):
                logger.warning(
                    "Repairing sector map for world='%s', sector='%s' from type %s to dict.",
                    world_name,
                    sector_name,
                    type(sector_map).__name__,
                )
                sector_map = {}
                world_map[sector_name] = sector_map

            existing = sector_map.setdefault(arena_name, [])
            # Deduplicate while preserving order
            for obj in objects:
                if obj and obj not in existing:
                    existing.append(obj)

            logger.debug(
                "Updated arena objects: world='%s', sector='%s', arena='%s', objects='%s'",
                world_name,
                sector_name,
                arena_name,
                ", ".join(existing),
            )
            logger.debug(
                "Post-update objects for world='%s', sector='%s', arena='%s': %s",
                world_name,
                sector_name,
                arena_name,
                existing,
            )
        except Exception as e:
            logger.error(
                "Error updating arena objects for world='%s', sector='%s', arena='%s': %s",
                world_name,
                sector_name,
                arena_name,
                e,
            )
