import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("bionic_hand"), "config", "bionic_hand.yaml"
    )

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the bionic_hand parameter YAML.",
    )

    hand_node = Node(
        package="bionic_hand",
        executable="bionic_hand_node",
        name="bionic_hand",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([params_arg, hand_node])
