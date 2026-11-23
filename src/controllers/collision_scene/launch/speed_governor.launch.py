"""Launch speed_governor (+ limiter_tuner, which applies its scale to
joint_command_limiter's max_velocity / max_acceleration).

Needs collision_scene_node running for /collision_scene/joint_distance.

    ros2 launch collision_scene speed_governor.launch.py               # governor + tuner
    ros2 launch collision_scene speed_governor.launch.py tuner:=false  # publish only
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("collision_scene")
    return LaunchDescription([
        DeclareLaunchArgument("tuner", default_value="true",
                              description="also run limiter_tuner"),
        Node(
            package="collision_scene",
            executable="speed_governor_node",
            name="speed_governor",
            output="screen",
            parameters=[os.path.join(share, "config", "speed_governor.yaml")],
        ),
        Node(
            package="collision_scene",
            executable="limiter_tuner_node",
            name="limiter_tuner",
            output="screen",
            parameters=[os.path.join(share, "config", "limiter_tuner.yaml")],
            condition=IfCondition(LaunchConfiguration("tuner")),
        ),
    ])
