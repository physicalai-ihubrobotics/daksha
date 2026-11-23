"""Launch the collision monitor.

    ros2 launch collusion collision_monitor.launch.py
    ros2 launch collusion collision_monitor.launch.py joint_states_topic:=/hw/joint_states
    ros2 launch collusion collision_monitor.launch.py geometry:=mesh

geometry:
  primitive  (default) robot_collision.urdf enclosing cylinders/boxes — fast (~1 ms),
             what the IK avoids; oversized so some pairs overlap at the zero pose.
  mesh       robot.urdf real STL collision meshes — accurate, ~190 ms per check (~4 Hz).

kinematics' joint_direction.yaml is loaded too, so angles published by
ik_node are mapped back to URDF convention before checking.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from collusion.pydeps import placo_env

GEOMETRY = {
    'primitive': ('robot_collision.urdf', 50.0),
    'mesh':      ('robot.urdf',           2.0),
}


def _monitor(context):
    geometry = LaunchConfiguration('geometry').perform(context)
    if geometry not in GEOMETRY:
        raise RuntimeError(f"geometry must be one of {list(GEOMETRY)}, got '{geometry}'")
    urdf_file, rate = GEOMETRY[geometry]

    description_share = get_package_share_directory('daksha_description_full_body')
    config = os.path.join(
        get_package_share_directory('collusion'), 'config', 'collision_monitor.yaml')
    joint_direction = os.path.join(
        get_package_share_directory('kinematics'), 'config', 'joint_direction.yaml')

    return [Node(
        package='collusion',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        additional_env=placo_env(),
        parameters=[
            config,
            joint_direction,
            {
                'joint_states_topic': LaunchConfiguration('joint_states_topic'),
                'urdf_file': os.path.join(description_share, 'urdf', urdf_file),
                'rate_hz': rate,
            },
        ],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('joint_states_topic', default_value='/joint_states'),
        DeclareLaunchArgument('geometry', default_value='primitive',
                              description='primitive | mesh'),
        OpaqueFunction(function=_monitor),
    ])
