"""Parsing of Qwen3-VL grounding replies (0-1000 grid -> pixels)."""

import pytest

from ros_agent.tracker import parse_box


def test_object_reply_scaled_to_pixels():
    box, label = parse_box('{"bbox_2d": [100, 250, 500, 750], "label": "cup"}', 640, 480)
    assert box == (64, 120, 320, 360) and label == "cup"


def test_list_reply_takes_first_and_tolerates_fences_and_think():
    reply = ('<think>hmm</think>```json\n[{"bbox_2d": [0, 0, 1000, 1000], "label": "a"},'
             ' {"bbox_2d": [1, 1, 2, 2], "label": "b"}]\n```')
    assert parse_box(reply, 640, 480) == ((0, 0, 640, 480), "a")


def test_swapped_corners_and_out_of_range_are_normalised():
    box, _ = parse_box('{"bbox_2d": [600, 900, 200, 1100]}', 1000, 1000)
    assert box == (200, 900, 600, 1000)


@pytest.mark.parametrize("reply", ['{"bbox_2d": null}', '{"bbox_2d": [1, 2, 3]}',
                                   '{"bbox_2d": [10, 10, 11, 11], "label": "dot"}'])
def test_not_found_or_degenerate(reply):
    assert parse_box(reply, 640, 480)[0] is None


def test_no_json_raises():
    with pytest.raises(ValueError):
        parse_box("I cannot see it", 640, 480)


from ros_agent.tracker import parse_boxes, pick_box  # noqa: E402


def test_parse_boxes_list():
    reply = '[{"bbox_2d": [0, 0, 500, 500], "label": "chair"}, {"bbox_2d": [500, 500, 1000, 1000], "label": "chair"}]'
    assert [b for b, _ in parse_boxes(reply, 100, 100)] == [(0, 0, 50, 50), (50, 50, 100, 100)]
    assert parse_boxes("[]", 100, 100) == []


def test_pick_box_prefers_box_under_point_then_nearest():
    a, b = ((0, 0, 50, 50), "chair"), ((60, 60, 100, 100), "chair")
    assert pick_box([a, b], (70, 70)) == b            # click inside b
    assert pick_box([a, b], (55, 20)) == a            # nearest centre
    assert pick_box([a, b], None) == a                # no reference: first
    big = ((0, 0, 100, 100), "table")
    assert pick_box([big, a], (10, 10)) == a          # smallest box containing the click
    assert pick_box([], (1, 1)) == (None, "")
