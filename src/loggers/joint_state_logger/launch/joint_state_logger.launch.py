import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    _log_dir = os.path.expanduser("~/daksha/logs")

    log_dir_arg = DeclareLaunchArgument(
        "log_dir",
        default_value=_log_dir,
        description="Directory that holds per-session Parquet folders (daksha_<ts>/)",
    )
    port_arg = DeclareLaunchArgument(
        "port",
        default_value="8140",
        description="Port for the built-in web control panel",
    )
    autostart_arg = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Begin recording a session on launch (else press Start in the UI)",
    )
    joint_hz_arg = DeclareLaunchArgument(
        "joint_states_max_hz",
        default_value="20.0",
        description="Throttle for the high-rate /joint_states stream (0 = log every message)",
    )

    logger_node = Node(
        package="joint_state_logger",
        executable="joint_state_logger",
        name="joint_state_logger",
        output="screen",
        parameters=[{
            "log_dir": LaunchConfiguration("log_dir"),
            "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
            "autostart": ParameterValue(LaunchConfiguration("autostart"), value_type=bool),
            "joint_states_max_hz": ParameterValue(
                LaunchConfiguration("joint_states_max_hz"), value_type=float),
        }],
    )

    return LaunchDescription([
        log_dir_arg,
        port_arg,
        autostart_arg,
        joint_hz_arg,
        logger_node,
    ])
