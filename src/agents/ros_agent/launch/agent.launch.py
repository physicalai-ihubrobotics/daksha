"""Launch ros_agent against the local Ollama server.

    ros2 launch ros_agent agent.launch.py                          # agent + Ollama qwen3.5:35b (one model for everything)
    ros2 launch ros_agent agent.launch.py llm_model:=qwen3.5:9b    # any `ollama list` model
    ros2 launch ros_agent agent.launch.py robot:=gen2 llm_url:=http://gpu-box:11434/v1
    ros2 launch ros_agent agent.launch.py llm_url:=                # tools only, no LLM
    ros2 launch ros_agent agent.launch.py web_port:=0              # no chat page
    ros2 launch ros_agent agent.launch.py pipeline:=false          # Q&A only, no robot tasks
    ros2 launch ros_agent agent.launch.py pipeline_execute:=true   # start the pipeline armed (default dry run)
    ros2 launch ros_agent agent.launch.py pipeline_model:=9b       # VLM for the pipeline (default from config)
    ros2 launch ros_agent agent.launch.py ik_debug:=true           # add embedded Placo IK viewer (dry run)

The model is served by Ollama (systemd service `ollama`), not by this package.
Pull models with `ollama pull <name>`; Ollama loads the model on the first
question, so the first answer takes longer.

One page, two tabs by default (three with IK debug): http://<robot>:8150/
  Ask          read-only Q&A with the world camera
  Robot tasks  real-robot VLM pick-and-place (settings: config/pipeline.yaml)
  IK Debug     live Placo/Viser viewer (launch with ik_debug:=true)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("ros_agent")
    aruco_share = get_package_share_directory("aruco_hand_eye")
    ik_debug = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(aruco_share, "launch", "aruco_reach.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("ik_debug")),
        launch_arguments={
            "execute": "false",
            "ui": "false",
            "viz": "true",
            "viz_port": "7000",
            "calib_ui": "false",
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="gen2"),
        DeclareLaunchArgument("llm_url", default_value="http://localhost:11434/v1",
                              description="Ollama OpenAI-compatible endpoint"),
        DeclareLaunchArgument("llm_model", default_value="qwen3.5:35b",
                              description="Ollama model name (see `ollama list`)"),
        DeclareLaunchArgument("web_port", default_value="8150"),
        DeclareLaunchArgument("pipeline", default_value="true",
                              description="real-robot VLM pipeline in the same process / page"),
        DeclareLaunchArgument("pipeline_execute", default_value="false",
                              description="start with real motion armed (default: dry run)"),
        DeclareLaunchArgument("pipeline_model", default_value="",
                              description="pipeline VLM (empty = config/pipeline.yaml)"),
        DeclareLaunchArgument("ik_debug", default_value="false",
                              description="start the Placo IK viewer in dry-run mode"),
        Node(
            package="ros_agent",
            executable="agent_node",
            name="ros_agent",
            output="screen",
            parameters=[
                os.path.join(share, "config", "agent.yaml"),
                {"robot_id": LaunchConfiguration("robot"),
                 "llm_base_url": LaunchConfiguration("llm_url"),
                 "llm_model": LaunchConfiguration("llm_model"),
                 "web_port": ParameterValue(LaunchConfiguration("web_port"), value_type=int),
                 "pipeline": ParameterValue(LaunchConfiguration("pipeline"), value_type=bool),
                 "pipeline_execute": ParameterValue(LaunchConfiguration("pipeline_execute"), value_type=bool),
                 "pipeline_model": LaunchConfiguration("pipeline_model")},
            ],
        ),
        ik_debug,
    ])
