import pytest
from core.tile import Tile
from core.agent_action import Event


def test_tile_event_and_collision():
    tile = Tile("world", "sector", "arena", "object", "spawn")
    event = Event("agent", "does", "something")

    tile.add_event(event)
    assert tile.has_event(event)

    tile.remove_event(event)
    assert not tile.has_event(event)

    assert not tile.is_collidable()
    tile.collision = True
    assert tile.is_collidable()
