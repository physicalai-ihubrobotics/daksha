import textwrap

import pytest

from ros_agent.robot_model import RobotModel

URDF = textwrap.dedent("""\
    <robot name="toy">
      <link name="base"><collision><geometry><box size="1 1 1"/></geometry></collision></link>
      <link name="l1"><collision><geometry><cylinder radius="0.1" length="0.5"/></geometry></collision></link>
      <link name="l2"/>
      <link name="tool"/>
      <link name="f1"/><link name="f2"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
        <limit lower="-1" upper="1" velocity="2" effort="10"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
        <limit lower="-2" upper="2" velocity="2" effort="10"/></joint>
      <joint name="tool_fixed" type="fixed"><parent link="l2"/><child link="tool"/></joint>
      <joint name="g1" type="prismatic"><parent link="tool"/><child link="f1"/>
        <limit lower="0" upper="0.04" velocity="1" effort="5"/></joint>
      <joint name="g2" type="prismatic"><parent link="tool"/><child link="f2"/>
        <limit lower="0" upper="0.04" velocity="1" effort="5"/>
        <mimic joint="g1" multiplier="1.0" offset="0"/></joint>
    </robot>
""")

SRDF = textwrap.dedent("""\
    <robot name="toy">
      <disable_collisions link1="base" link2="l1" reason="Adjacent"/>
      <disable_collisions link1="base" link2="l2" reason="Never"/>
    </robot>
""")


@pytest.fixture
def model(tmp_path):
    u, s = tmp_path / "r.urdf", tmp_path / "r.srdf"
    u.write_text(URDF)
    s.write_text(SRDF)
    return RobotModel(str(u), str(s), groups={"arm": {"base": "base", "tip": "tool"}},
                      end_effectors={"main": "tool"})


def test_structure(model):
    assert model.root == "base"
    assert model.actuated_joints() == ["j1", "j2", "g1"]
    assert model.groups["arm"]["joints"] == ["j1", "j2"]
    assert model.joint_link_owner("tool") == "j2"
    assert model.joint_link_owner("base") is None


def test_describe(model):
    d = model.describe()
    assert d["dof"] == 3
    assert d["mimic_joints"] == {"g2": {"joint": "g1", "multiplier": 1.0, "offset": 0.0}}
    assert d["links_without_collision"] == ["f1", "f2", "l2", "tool"]
    assert d["srdf_disabled_pairs"] == {"Adjacent": 1, "Never": 1}
    assert d["joint_limits"]["j2"]["upper"] == 2.0


def test_chain_errors(model):
    with pytest.raises(ValueError):
        model.chain_joints("l2", "l1")
    with pytest.raises(KeyError):
        model.chain_joints("base", "nope")
