"""Launch ros_agent together with the ArUco reach node.

The reach node exposes /aruco_reach/go_pose for external detections. It uses
IK and the trajectory/collision checks before publishing /joint_cmd. Motion is
dry-run by default; pass reach_execute:=true to arm it at startup.

    ros2 launch ros_agent agent_with_aruco_reach.launch.py
    ros2 launch ros_agent agent_with_aruco_reach.launch.py reach_ui:=false
    ros2 launch ros_agent agent_with_aruco_reach.launch.py reach_execute:=true
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    aruco_share = get_package_share_directory("aruco_hand_eye")
    agent_share = get_package_share_directory("ros_agent")
    reach_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{aruco_share}/launch/aruco_reach.launch.py"
        ),
        launch_arguments={
            "execute": LaunchConfiguration("reach_execute"),
            "ui": LaunchConfiguration("reach_ui"),
            "port": LaunchConfiguration("reach_port"),
            "calib_ui": LaunchConfiguration("calib_ui"),
        }.items(),
    )

    agent_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{agent_share}/launch/agent.launch.py"
        ),
        launch_arguments={
            "robot": LaunchConfiguration("robot"),
            "llm_url": LaunchConfiguration("llm_url"),
            "llm_model": LaunchConfiguration("llm_model"),
            "web_port": LaunchConfiguration("web_port"),
            "pipeline": LaunchConfiguration("pipeline"),
            "pipeline_execute": LaunchConfiguration("pipeline_execute"),
            "pipeline_model": LaunchConfiguration("pipeline_model"),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="gen2"),
        DeclareLaunchArgument("llm_url", default_value="http://localhost:11434/v1"),
        DeclareLaunchArgument("llm_model", default_value="qwen3.5:35b"),
        DeclareLaunchArgument("web_port", default_value="8150"),
        DeclareLaunchArgument("pipeline", default_value="true"),
        DeclareLaunchArgument("pipeline_execute", default_value="false"),
        DeclareLaunchArgument("pipeline_model", default_value=""),
        DeclareLaunchArgument(
            "reach_execute", default_value="false",
            description="Arm real robot motion in aruco_reach (default: dry run)",
        ),
        DeclareLaunchArgument("reach_ui", default_value="true"),
        DeclareLaunchArgument("reach_port", default_value="8171"),
        DeclareLaunchArgument(
            "calib_ui", default_value="false",
            description="Also start the hand-eye calibration UI",
        ),
        agent_launch,
        reach_launch,
    ])
