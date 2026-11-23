"""Message -> JSON conversion used by the generic topic tools."""

import array

import pytest

from ros_agent.topics import _flatten, fit, select_field, to_plain


def test_small_arrays_kept_big_arrays_summarised():
    assert to_plain([1, 2, 3]) == [1, 2, 3]
    big = to_plain(array.array("B", range(200)), preview=4)
    assert big["_array_length"] == 200 and big["first"] == [0, 1, 2, 3]
    assert big["min"] == 0 and big["max"] == 199
    assert to_plain(float("nan")) == "nan"


def test_select_field_paths():
    data = {"pose": {"position": {"x": 1.5}}, "position": [0.1, 0.2, 0.3]}
    assert select_field(data, "pose.position.x") == 1.5
    assert select_field(data, "position[2]") == 0.3
    with pytest.raises(KeyError):
        select_field(data, "pose.orientation")


def test_flatten_skips_stamps_and_summaries():
    data = {"header": {"stamp": {"sec": 5}, "frame_id": "base"},
            "name": ["a"], "position": [0.5], "data": {"_array_length": 99}}
    assert _flatten(data) == {"header.frame_id": "base", "name[0]": "a", "position[0]": 0.5}


def test_fit_shrinks_then_truncates():
    data, cut = fit({"x": list(range(10))}, lambda p: {"x": list(range(10))})
    assert not cut
    huge = {"s": "y" * 10000}
    data, cut = fit(huge, lambda p: huge, budget=100)
    assert cut and len(data["truncated_json"]) < 120
