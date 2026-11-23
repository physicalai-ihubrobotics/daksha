"""Launch collision_scene node (+ optional RViz).

URDF is read from /robot_description (published by robot_state_publisher).
Joint states are read from /joint_states.

    # Typical use — URDF from topic, RViz opens automatically
    ros2 launch collision_scene collision_scene.launch.py

    # Without RViz
    ros2 launch collision_scene collision_scene.launch.py rviz:=false

    # Mesh geometry (accurate, ~2 Hz)
    ros2 launch collision_scene collision_scene.launch.py geometry:=mesh

    # Different joint states topic
    ros2 launch collision_scene collision_scene.launch.py joint_states_topic:=/hw/joint_states

    # URDF from file instead of topic
    ros2 launch collision_scene collision_scene.launch.py urdf_source:=file

RViz config: config/collision_scene.rviz
  Displays: Grid | RobotModel (/robot_description) | MarkerArray (/collision_scene/markers)
  Fixed frame: world
  Red spheres  = links in contact now  (reactive)
  Yellow spheres + line = links about to collide (predictive)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

GEOMETRY = {
    "primitive": 30.0,   # C++ batch → ~1-2 ms/check
    "mesh":       2.0,   # ~190 ms/check
}


def _node(context):
    geometry = LaunchConfiguration("geometry").perform(context)
    if geometry not in GEOMETRY:
        raise RuntimeError(
            f"geometry must be one of {list(GEOMETRY)}, got '{geometry}'")

    urdf_source = LaunchConfiguration("urdf_source").perform(context)
    scene_share = get_package_share_directory("collision_scene")
    joint_direction = os.path.join(
        get_package_share_directory("kinematics"), "config", "joint_direction.yaml")

    extra: dict = {
        "rate_hz": GEOMETRY[geometry],
        "urdf_source": urdf_source,
        "joint_states_topic": LaunchConfiguration("joint_states_topic"),
    }

    if urdf_source == "file":
        description_share = get_package_share_directory("daksha_description_full_body")
        urdf_file = {
            "primitive": "robot_collision.urdf",
            "mesh":      "robot.urdf",
        }[geometry]
        extra["urdf_file"] = os.path.join(description_share, "urdf", urdf_file)

    nodes = [Node(
        package="collision_scene",
        executable="collision_scene_node",
        name="collision_scene",
        output="screen",
        parameters=[
            os.path.join(scene_share, "config", "collision_scene.yaml"),
            joint_direction,
            extra,
        ],
    )]

    if LaunchConfiguration("rviz").perform(context).lower() in ("true", "1"):
        nodes.append(Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=[
                "-d", os.path.join(scene_share, "config", "collision_scene.rviz"),
            ],
            output="screen",
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
        DeclareLaunchArgument("geometry", default_value="primitive",
                              description="primitive | mesh  (affects rate_hz)"),
        DeclareLaunchArgument("urdf_source", default_value="topic",
                              description="topic (from /robot_description) | file"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="true | false — open RViz with collision_scene.rviz"),
        OpaqueFunction(function=_node),
    ])
