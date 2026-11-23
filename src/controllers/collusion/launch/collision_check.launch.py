"""Interactive URDF collision check: drag joints, watch for collisions.

    ros2 launch collusion collision_check.launch.py
    ros2 launch collusion collision_check.launch.py geometry:=primitive
    ros2 launch collusion collision_check.launch.py rviz:=false

Starts robot_state_publisher (robot.urdf), joint_state_publisher_gui, the
collision monitor and RViz. In RViz, red spheres mark links in contact
(reactive) and yellow spheres + line mark pairs about to collide
(predictive); new contacts are logged.

Per joint:
    ros2 topic echo /collision/predictive_guard   # 1 = going to collide
    ros2 topic echo /collision/reactive_guard     # 1 = has collided
    ros2 topic echo /collision/joint_distance     # closest distance (m)

geometry:
  mesh       (default) robot.urdf's real STL collision meshes - what RViz
             shows, accurate, ~190 ms per check so it runs at ~4 Hz.
  primitive  robot_collision.urdf's enclosing cylinders/boxes - what the IK
             avoids, ~1 ms per check, but oversized: several pairs already
             overlap by up to ~22 mm at the all-zero pose.

Contact defaults to 0 mm and the predictive margin to 10 mm: at rest, links
two joints apart are only 15-20 mm apart on the real meshes.

The GUI publishes URDF-convention angles, so kinematics' joint_direction.yaml
is deliberately NOT loaded here (unlike collision_monitor.launch.py).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from collusion.pydeps import placo_env

GEOMETRY = {
    # urdf file, check rate (Hz). Mesh: ~190 ms per check, two checks per
    # tick while joints move (current + predicted pose).
    'mesh': ('robot.urdf', 2.0),
    'primitive': ('robot_collision.urdf', 30.0),
}


def _monitor(context):
    geometry = LaunchConfiguration('geometry').perform(context)
    if geometry not in GEOMETRY:
        raise RuntimeError(f"geometry must be one of {list(GEOMETRY)}, got '{geometry}'")
    urdf, rate = GEOMETRY[geometry]
    description_share = get_package_share_directory('daksha_description_full_body')
    collusion_share = get_package_share_directory('collusion')
    arg = lambda name: float(LaunchConfiguration(name).perform(context))  # noqa: E731

    return [Node(
        package='collusion',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        additional_env=placo_env(),
        parameters=[
            os.path.join(collusion_share, 'config', 'collision_monitor.yaml'),
            {
                'urdf_file': os.path.join(description_share, 'urdf', urdf),
                'rate_hz': rate,
                'joint_states_topic': '/joint_states',
                'contact_distance': arg('contact_distance'),
                'predictive_distance': arg('predictive_distance'),
                'prediction_horizon': arg('prediction_horizon'),
                'joint_state_timeout': 1.0,
            },
        ],
    )]


def generate_launch_description():
    description_share = get_package_share_directory('daksha_description_full_body')
    collusion_share = get_package_share_directory('collusion')

    with open(os.path.join(description_share, 'urdf', 'robot.urdf')) as f:
        robot_urdf = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('geometry', default_value='mesh',
                              description='mesh | primitive'),
        DeclareLaunchArgument('contact_distance', default_value='0.0'),
        DeclareLaunchArgument('predictive_distance', default_value='0.01'),
        DeclareLaunchArgument('prediction_horizon', default_value='0.5'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_urdf}],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
        ),
        OpaqueFunction(function=_monitor),
        Node(
            condition=IfCondition(LaunchConfiguration('rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(collusion_share, 'config', 'collision_check.rviz')],
            output='screen',
        ),
    ])
