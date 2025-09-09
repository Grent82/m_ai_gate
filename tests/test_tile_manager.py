import pytest
from core.tile_manager import TileManager
from core.agent_action import Event


def test_get_tile_and_bounds():
    tm = TileManager(2, 2, "world")
    assert tm.is_within_bounds(1, 1)
    assert not tm.is_within_bounds(-1, 0)
    assert not tm.is_within_bounds(2, 2)

    tile = tm.get_tile(0, 0)
    assert tile.world_name == "world"

    with pytest.raises(ValueError):
        tm.get_tile(5, 5)


def test_sector_and_path():
    tm = TileManager(3, 3, "world")
    tm.set_sector(1, 1, "S1")
    tm.set_arena(1, 1, "A1")
    tm.set_game_object(1, 1, "O1")

    assert tm.get_current_sector((1, 1)) == "S1"

    with pytest.raises(ValueError):
        tm.get_current_sector((0, 0))

    assert tm.get_tile_path((1, 1), "arena") == "world:S1:A1"

    with pytest.raises(ValueError):
        tm.get_tile_path((1, 1), "invalid")

    tm2 = TileManager(1, 1, "world")
    with pytest.raises(ValueError):
        tm2.get_tile_path((0, 0), "sector")


def test_nearby_tiles_and_events():
    tm = TileManager(3, 3, "world")
    positions = tm.get_nearby_tiles_positions((1, 1), 1)
    assert len(positions) == 9

    event = Event("hero", "meets", "villain")
    tm.add_event_to_tile(0, 0, event)
    assert tm.get_tile(0, 0).has_event(event)
    tm.remove_event_from_tile(0, 0, event)
    assert not tm.get_tile(0, 0).has_event(event)

    tm.set_collision(0, 0, True)
    assert tm.is_collidable(0, 0)
