"""
hardware.launch.py — Daksha hardware bringup.

Brings up:
  - CAN interface (500k bitrate)
  - URDF / robot_state_publisher (daksha_description)
  - ros2_control (daksha_hardware plugin via daksha_description bringup)
  - LiDAR + bumper STM32 board (daksha_lidar: stm32_sensor_mux + sllidar)
  - Bumper safety node (daksha_bumper)
  - IMU serial bridge + static TF
  - EKF (robot_localization)
  - Scan filter (daksha_bringup)
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Kill stale hardware processes from previous runs
    cleanup_cmd = ExecuteProcess(
        cmd=['bash', '-c',
             'pkill -f "[s]llidar_node" 2>/dev/null; '
             'pkill -f "[s]tm32_sensor_mux" 2>/dev/null; '
             'pkill -f "[b]umper_node" 2>/dev/null; '
             'pkill -f "[s]can_filter" 2>/dev/null; '
             'pkill -f "[r]os2_control_node" 2>/dev/null; '
             'pkill -f "[r]obot_state_publisher" 2>/dev/null; '
             'pkill -f "[i]mu_serial_bridge" 2>/dev/null; '
             'sleep 1; echo "[cleanup] Stale hardware processes cleared"; exit 0'],
        output='screen'
    )

    lidar_launch = os.path.join(
        get_package_share_directory('daksha_lidar'),
        'launch',
        'sllidar_s2_launch.py'
    )

    # Reads /tmp/amr_bumper, created by stm32_sensor_mux in the lidar launch;
    # it retries until the port exists (and holds the e-stop meanwhile)
    bumper_launch = os.path.join(
        get_package_share_directory('daksha_bumper'),
        'launch',
        'bumper.launch.py'
    )

    description_launch_path = os.path.join(
        get_package_share_directory('daksha_description'),
        'launch',
        'bringup.launch.py'
    )

    setup_can = ExecuteProcess(
        cmd=[
            'bash', '-c',
            "if ! ip link show can0 2>/dev/null | grep -q 'state UP'; then "
            'echo 123456 | sudo -S ip link set can0 type can bitrate 500000 2>/dev/null; '
            'echo 123456 | sudo -S ip link set can0 up 2>/dev/null || true; '
            'fi'
        ],
        output='screen'
    )

    ekf_config_file = os.path.join(
        get_package_share_directory('daksha_bringup'),
        'config',
        'ekf.yaml'
    )

    imu_bridge_node = Node(
        package='daksha_bringup',
        executable='imu_serial_bridge',
        name='imu_serial_bridge',
        output='screen',
        # /dev/imu_esp32 is created by /etc/udev/rules.d/99-amr-serial.rules
        # (CP2102 10c4:ea60). Leave empty to fall back to VID/PID auto-scan.
        parameters=[{'port': '/dev/imu_esp32'}],
    )

    static_imu_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_imu_tf_publisher',
        output='screen',
        arguments=[
            '--x', '0.03', '--y', '0', '--z', '0.38',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base_link', '--child-frame-id', 'imu_link'
        ]
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_file]
    )

    scan_filter_node = Node(
        package='daksha_bringup',
        executable='scan_filter',
        name='scan_filter_node',
        output='screen',
    )

    # imu_filter_node = Node(
    #     package='imu_filter_madgwick',
    #     executable='imu_filter_madgwick_node',
    #     name='imu_filter',
    #     output='screen',
    #     parameters=[{
    #         'use_mag': False,
    #         'publish_tf': False,
    #         'imu_frame': 'imu_link',
    #         'world_frame': 'enu',
    #         'gain': 0.1,
    #         'orientation_stddev': 0.05,
    #     }],
    # )

    return LaunchDescription([
        # 1. Kill stale processes first
        cleanup_cmd,
        # 2. Wait for cleanup, then launch everything
        TimerAction(
            period=2.0,
            actions=[
                setup_can,
                IncludeLaunchDescription(PythonLaunchDescriptionSource(description_launch_path)),
                IncludeLaunchDescription(PythonLaunchDescriptionSource(lidar_launch)),
                IncludeLaunchDescription(PythonLaunchDescriptionSource(bumper_launch)),
                scan_filter_node,
                imu_bridge_node,
                static_imu_tf_publisher,
                ekf_node,
                # imu_filter_node,
            ]
        ),
    ])
