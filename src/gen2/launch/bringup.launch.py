from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from pathlib import Path
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    vcan_bridge_node_py = Node(
        package="gen2",
        executable="vcan_bridge",
        output="screen",
    )

    gravity_torque_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("dynamics"),
                "launch",
                "gravity_torque.launch.py",
            )
        )
    )
    start_all_cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("camera_launch"),
                "launch",
                "start_all_cameras.launch.py",
            )
        )
    )
    gesture_management_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gesture_management"),
                "launch",
                "gesture_management.launch.py",
            )
        )
    )
    joint_analyzer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("joint_analyzer"),
                "launch",
                "joint_analyzer.launch.py",
            )
        )
    )
    joint_cmd_logger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("joint_cmd_logger"),
                "launch",
                "joint_cmd_logger.launch.py",
            )
        )
    )
    client_ui_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("daksha_ui"),
                "launch",
                "client_ui.launch.py",
            )
        )
    )
    vajara_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("vajara"),
                "launch",
                "vajara.launch.py",
            )
        )
    )

    urdf = Path(
        os.path.join(
            get_package_share_directory("daksha_description_full_body"),
            "urdf",
            "robot.urdf",
        )
    ).read_text()

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            os.path.join(
                get_package_share_directory("gen2"), "config", "controllers.yaml"
            ),
        ],
        # Jazzy: controller_manager reads the URDF from robot_state_publisher's topic.
        remappings=[("~/robot_description", "/robot_description")],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": urdf}],
        output="screen",
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )
    left_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_mit_controller"],
        output="screen",
    )

    right_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_mit_controller"],
        output="screen",
    )

    left_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_gripper_mit_controller"],
        output="screen",
    )

    right_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_gripper_mit_controller"],
        output="screen",
    )



    # Hardware joint-direction corrections (e.g. right_joint_6, wired opposite
    # to the URDF) live in vr_teleop/config/joint_overrides.yaml so they sit
    # next to the teleop config. The limiter applies them at the hardware
    # boundary only; placo/IK stay in URDF convention.
    joint_overrides_yaml = os.path.join(
        get_package_share_directory("vr_teleop"), "config", "joint_overrides.yaml"
    )
    joint_command_node = Node(
        package="gen2",
        executable="joint_cmd",
        output="screen",
        parameters=[joint_overrides_yaml, {"enforce_joint_limits": False}],
    )
    GravityToJointCmd_node = Node(
        package="gen2",
        executable="GravityToJointCmd",
        output="screen",
    )
    joint_cmd_publisher_from_joint_sates_node = Node(
        package="gen2",
        executable="joint_cmd_publisher_from_joint_sates",
        output="screen",
        parameters=[{"publish_joint_cmd": False}],
    )
    teach_mode_node = Node(
        package="gen2",
        executable="teach_mode_node",
        output="screen",
        parameters=[{"teach_mode": True}],
    )
    gravity_scale_setter_node = Node(
        package="gen2",
        executable="gravity_scale_setter",
        output="screen",
    )
    mode_toggler_node = Node(
        package="gen2",
        executable="mode_toggler",
        output="screen",
        parameters=[{"mode": "teach"}],
    )
    vr_management_ui_node = Node(
        package="vr_teleop",
        executable="vr_management_ui",
        output="screen",
        parameters=[{"enabled": False}],
    )
    vr_pose_relay_node = Node(
        package="vr_teleop",
        executable="vr_pose_relay",
        output="screen",
    )
    ik_node = Node(
        package="kinematics",
        executable="ik_node",
        output="screen",
        parameters=[
            os.path.join(
                get_package_share_directory("kinematics"), "config", "orientation.yaml"
            ),
            os.path.join(
                get_package_share_directory("kinematics"), "config", "joint_limits.yaml"
            ),
        ],
    )

    battery_info_node = Node(
        package="gen2",
        executable="battery_info",
        output="screen",
    )
    home_move_service_node = Node(
        package="gen2",
        executable="HomeMoveService",
        output="screen",
    )

    leader_controller_ui_node = Node(
        package="gen2_leader",
        executable="leader_controller_ui",
        output="screen",
    )

    arm_recovery_watchdog_node = Node(
        package="gen2",
        executable="arm_recovery_watchdog",
        output="screen",
    )

    return LaunchDescription([
        vcan_bridge_node_py,
        TimerAction(
            period=3.0,
            actions=[
                gravity_torque_launch,
                start_all_cameras_launch,
                # gesture_management_launch,
                # joint_analyzer_launch,
                # client_ui_launch,
                # vajara_launch,
                controller_manager,
                robot_state_publisher,
                RegisterEventHandler(
                    OnProcessStart(
                        target_action=controller_manager,
                        on_start=[
                            joint_state_broadcaster,
                            left_arm_controller,
                            right_arm_controller,
                            left_gripper_controller,
                            right_gripper_controller,
                            joint_command_node,
                            GravityToJointCmd_node,
                            joint_cmd_publisher_from_joint_sates_node,
                            gravity_scale_setter_node,
                            teach_mode_node,
                            mode_toggler_node,
                            # vr_management_ui_node,
                            # ik_node,
                            battery_info_node,
                            home_move_service_node,
                            # leader_controller_ui_node,
                            # arm_recovery_watchdog_node,
                        ],
                    )
                ),
            ],
        ),
    ])