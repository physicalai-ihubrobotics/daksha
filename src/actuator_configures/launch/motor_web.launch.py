from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Real CAN on the PEAK PCAN-USB Pro FD (driver: peak_usb). The interfaces are
# brought up at 1 Mbit/s automatically by robot_setup/scripts/install_pcan.sh;
# check names with `install_pcan.sh --status` if the onboard mttcan ports load.
def generate_launch_description():
    host_arg = DeclareLaunchArgument(
        "host",
        default_value="::",
        description="Address to serve on; 127.0.0.1 for local-only access.",
    )
    port_arg = DeclareLaunchArgument(
        "port",
        default_value="8160",
        description="HTTP port of the web UI.",
    )
    right_channel_arg = DeclareLaunchArgument(
        "right_channel",
        default_value="can0",
        description="SocketCAN interface for the right arm (PEAK channel 1).",
    )
    left_channel_arg = DeclareLaunchArgument(
        "left_channel",
        default_value="can1",
        description="SocketCAN interface for the left arm (PEAK channel 2).",
    )

    motor_web_node = Node(
        package="actuator_configures",
        executable="motor_web",
        output="screen",
        parameters=[{
            "right_channel": ParameterValue(LaunchConfiguration("right_channel"), value_type=str),
            "left_channel": ParameterValue(LaunchConfiguration("left_channel"), value_type=str),
            "host": ParameterValue(LaunchConfiguration("host"), value_type=str),
            "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
        }],
    )

    return LaunchDescription([
        host_arg,
        port_arg,
        right_channel_arg,
        left_channel_arg,
        motor_web_node,
    ])
