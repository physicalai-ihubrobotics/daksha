#!/usr/bin/env python3
"""ROS2 node for controlling differential drive robot with CANopen motors and IMU yaw."""

import json
import math
import time
from dataclasses import dataclass
import threading

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Bool, Float32, String

try:
    from amr_auto.cmd_vel import (
        MotorError,
        SocketCanBus,
        CanOpenNode,
        Motor,
        TwoMotorController,
        clamp,
        rpm_to_dec
    )
except ImportError:
    from cmd_vel import (
        MotorError,
        SocketCanBus,
        CanOpenNode,
        Motor,
        TwoMotorController,
        clamp,
        rpm_to_dec
    )


class DifferentialDriveController(Node):
    """ROS2 node for controlling differential drive robot with CANopen motors."""
    
    def __init__(self):
        super().__init__('differential_drive_controller')
        

        # Declare all parameters first
        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('left_node_id', 1)
        self.declare_parameter('right_node_id', 2)
        self.declare_parameter('wheel_circumference', 0.575)  #0.248
        self.declare_parameter('wheel_distance',  0.50)
        self.declare_parameter('ticks_per_revolution', 4096)
        self.declare_parameter('max_linear_velocity', 1.0)
        self.declare_parameter('max_angular_velocity', 0.8)
        self.declare_parameter('max_rpm', 30)
        self.declare_parameter('max_linear_velocity_topic', '/max_linear_velocity')
        self.declare_parameter('max_angular_velocity_topic', '/max_angular_velocity')
        self.declare_parameter('max_rpm_topic', '/max_rpm')
        self.declare_parameter('emergency_stop_topic', '/emergency_stop')
        self.declare_parameter('publish_rate_topic', '/publish_rate')
        self.declare_parameter('motor_command_timeout_topic', '/motor_command_timeout')
        self.declare_parameter('balance_enabled_topic', '/balance_enabled')
        self.declare_parameter('balance_kp_topic', '/balance_kp')
        self.declare_parameter('max_correction_rpm_topic', '/max_correction_rpm')
        self.declare_parameter('safe_balance_ratio_topic', '/safe_balance_ratio')
        self.declare_parameter('use_imu_yaw_topic', '/use_imu_yaw')
        self.declare_parameter('imu_yaw_offset_topic', '/imu_yaw_offset')
        self.declare_parameter('mode_topic', '/mode')
        self.declare_parameter('timeout_sec_topic', '/timeout_sec')
        self.declare_parameter('ramp_enabled_topic', '/ramp_enabled')
        self.declare_parameter('accel_rpm_per_sec_topic', '/accel_rpm_per_sec')
        self.declare_parameter('decel_rpm_per_sec_topic', '/decel_rpm_per_sec')
        self.declare_parameter('runtime_config_topic', '/runtime_config')
        self.declare_parameter('runtime_config_publish_rate', 1.0)
        self.declare_parameter('reset_runtime_config_topic', '/reset_runtime_config')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('motor_command_timeout', 0.5)
        self.declare_parameter('balance_enabled', False)
        self.declare_parameter('balance_kp', 0.002)
        self.declare_parameter('max_correction_rpm', 20)
        self.declare_parameter('safe_balance_ratio', 0.15)
        
        # IMU parameters
        self.declare_parameter('use_imu_yaw',  True)  # Use IMU instead of encoder yaw
        self.declare_parameter('imu_topic', '/bno055/imu')  # IMU topic
        self.declare_parameter('imu_yaw_offset', 0.0)  # Initial yaw offset (radians)
        self.declare_parameter('imu_yaw_timeout', 0.30)
        self.declare_parameter('use_imu_gyro_extrapolation', True)
        self.declare_parameter('max_imu_extrapolation', 0.12)
        self.declare_parameter('invert_imu_yaw', False)
        
        # Motor inversion parameters
        self.declare_parameter('invert_left', False)
        self.declare_parameter('invert_right', True)
        # Odometry Covariances
        self.declare_parameter('odom_pose_covariance', [0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                        0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
                                                        0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
                                                        0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
                                                        0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
                                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
        self.declare_parameter('odom_twist_covariance', [0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
                                                         0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
                                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
        
        # ODOMETRY CORRECTION PARAMETERS
        self.declare_parameter('swap_odometry_wheels', True)  # Swap left/right for odometry
        self.declare_parameter('invert_odometry_angular', False)  # Invert angular in odometry
        self.declare_parameter('invert_odometry_linear', False)  # Invert linear in odometry
        self.declare_parameter('invert_odometry_left', True)  # Fix mirrored encoder sign after wheel swap
        self.declare_parameter('invert_odometry_right', False)
        
        # NEW: Swap angular and linear interpretation in odometry
        # When True: wheel difference becomes linear motion, wheel sum becomes rotation
        self.declare_parameter('swap_angular_linear_odometry', False)
        
        # CMD_VEL MAPPING PARAMETERS
        self.declare_parameter('swap_cmd_vel_wheels', False)  # Swap left/right wheel mapping
        self.declare_parameter('invert_cmd_vel_angular', True)  # Invert angular velocity direction
        self.declare_parameter('invert_cmd_vel_linear', False)  # Invert linear velocity direction
        
        # NEW: Swap angular and linear interpretation in cmd_vel
        # When True: linear command becomes rotation, rotation command becomes linear
        self.declare_parameter('swap_angular_linear_cmd_vel', False)
        
        self.declare_parameter('precise_speed_command', True)
        self.declare_parameter('nmt_start', True)
        self.declare_parameter('fix_din3', True)
        self.declare_parameter('mode', 3)
        self.declare_parameter('timeout_sec', 0.35)
        self.declare_parameter('retries', 3)
        self.declare_parameter('debug_frames', False)
        self.declare_parameter('keep_enabled', False)

        # Smooth acceleration/deceleration (RPM/s)
        # Limits how fast the *sent* motor RPM can change (prevents sudden start/stop).
        self.declare_parameter('ramp_enabled', True)
        self.declare_parameter('accel_rpm_per_sec', 0.05)
        self.declare_parameter('decel_rpm_per_sec', 200.0)
        
        # Get parameters
        self.can_interface = self.get_parameter('can_interface').value
        self.left_node_id = self.get_parameter('left_node_id').value
        self.right_node_id = self.get_parameter('right_node_id').value
        self.wheel_circumference = self.get_parameter('wheel_circumference').value
        self.wheel_distance = self.get_parameter('wheel_distance').value
        self.ticks_per_rev = self.get_parameter('ticks_per_revolution').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.max_rpm = self.get_parameter('max_rpm').value
        self.max_linear_velocity_topic = self.get_parameter('max_linear_velocity_topic').value
        self.max_angular_velocity_topic = self.get_parameter('max_angular_velocity_topic').value
        self.max_rpm_topic = self.get_parameter('max_rpm_topic').value
        self.emergency_stop_topic = self.get_parameter('emergency_stop_topic').value
        self.publish_rate_topic = self.get_parameter('publish_rate_topic').value
        self.motor_command_timeout_topic = self.get_parameter('motor_command_timeout_topic').value
        self.balance_enabled_topic = self.get_parameter('balance_enabled_topic').value
        self.balance_kp_topic = self.get_parameter('balance_kp_topic').value
        self.max_correction_rpm_topic = self.get_parameter('max_correction_rpm_topic').value
        self.safe_balance_ratio_topic = self.get_parameter('safe_balance_ratio_topic').value
        self.use_imu_yaw_topic = self.get_parameter('use_imu_yaw_topic').value
        self.imu_yaw_offset_topic = self.get_parameter('imu_yaw_offset_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value
        self.timeout_sec_topic = self.get_parameter('timeout_sec_topic').value
        self.ramp_enabled_topic = self.get_parameter('ramp_enabled_topic').value
        self.accel_rpm_per_sec_topic = self.get_parameter('accel_rpm_per_sec_topic').value
        self.decel_rpm_per_sec_topic = self.get_parameter('decel_rpm_per_sec_topic').value
        self.runtime_config_topic = self.get_parameter('runtime_config_topic').value
        self.runtime_config_publish_rate = float(self.get_parameter('runtime_config_publish_rate').value)
        self.reset_runtime_config_topic = self.get_parameter('reset_runtime_config_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.motor_command_timeout = self.get_parameter('motor_command_timeout').value
        
        # IMU parameters
        self.use_imu_yaw = self.get_parameter('use_imu_yaw').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.imu_yaw_offset = self.get_parameter('imu_yaw_offset').value
        self.imu_yaw_timeout = float(self.get_parameter('imu_yaw_timeout').value)
        self.use_imu_gyro_extrapolation = bool(self.get_parameter('use_imu_gyro_extrapolation').value)
        self.max_imu_extrapolation = float(self.get_parameter('max_imu_extrapolation').value)
        self.invert_imu_yaw = bool(self.get_parameter('invert_imu_yaw').value)
        
        # Get odometry correction parameters
        self.swap_odometry_wheels = self.get_parameter('swap_odometry_wheels').value
        self.invert_odometry_angular = self.get_parameter('invert_odometry_angular').value
        self.invert_odometry_linear = self.get_parameter('invert_odometry_linear').value
        self.invert_odometry_left = self.get_parameter('invert_odometry_left').value
        self.invert_odometry_right = self.get_parameter('invert_odometry_right').value
        self.swap_angular_linear_odometry = self.get_parameter('swap_angular_linear_odometry').value
        self.odom_pose_covariance = self.get_parameter('odom_pose_covariance').value
        self.odom_twist_covariance = self.get_parameter('odom_twist_covariance').value
        
        # Get cmd_vel mapping parameters
        self.swap_cmd_vel_wheels = self.get_parameter('swap_cmd_vel_wheels').value
        self.invert_cmd_vel_angular = self.get_parameter('invert_cmd_vel_angular').value
        self.invert_cmd_vel_linear = self.get_parameter('invert_cmd_vel_linear').value
        self.swap_angular_linear_cmd_vel = self.get_parameter('swap_angular_linear_cmd_vel').value
        
        # Initialize CAN bus and motors
        self.io = SocketCanBus(interface=self.can_interface, 
                               debug=self.get_parameter('debug_frames').value)
        
        left_motor = Motor(
            CanOpenNode(self.io, node_id=self.left_node_id,
                       timeout=self.get_parameter('timeout_sec').value,
                       retries=self.get_parameter('retries').value,
                       debug=self.get_parameter('debug_frames').value)
        )
        right_motor = Motor(
            CanOpenNode(self.io, node_id=self.right_node_id,
                       timeout=self.get_parameter('timeout_sec').value,
                       retries=self.get_parameter('retries').value,
                       debug=self.get_parameter('debug_frames').value)
        )
        
        self.controller = TwoMotorController(
            left=left_motor,
            right=right_motor,
            invert_left=self.get_parameter('invert_left').value,
            invert_right=self.get_parameter('invert_right').value,
        )
        
        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Yaw angle in radians
        
        # IMU state
        self.imu_theta = 0.0
        self.imu_angular_velocity_z = 0.0
        self.imu_stamp_sec = None
        self.imu_available = False
        self.imu_lock = threading.Lock()
        
        # Encoder state
        self.last_left_ticks = 0
        self.last_right_ticks = 0
        self.last_time = self.get_ros_time_sec()
        
        # Motor command state
        self.target_left_rpm = 0
        self.target_right_rpm = 0
        self.last_motor_command_time = time.time()
        self.emergency_stop_active = False
        self.ramp_enabled = bool(self.get_parameter('ramp_enabled').value)
        self.accel_rpm_per_sec = float(self.get_parameter('accel_rpm_per_sec').value)
        self.decel_rpm_per_sec = float(self.get_parameter('decel_rpm_per_sec').value)
        self.sent_left_rpm = 0
        self.sent_right_rpm = 0
        self.last_ramp_time = time.time()

        # Measured wheel speed (from encoders)
        self.current_left_rpm = 0.0
        self.current_right_rpm = 0.0
        self.default_runtime_config = {
            'max_linear_velocity': 1.0,
            'max_angular_velocity': 0.8,
            'max_rpm': 10,
            'publish_rate': 50.0,
            'motor_command_timeout': 0.5,
            'balance_enabled': False,
            'balance_kp': 0.002,
            'max_correction_rpm': 20,
            'safe_balance_ratio': 0.15,
            'use_imu_yaw': True,
            'imu_yaw_offset': 0.0,
            'imu_yaw_timeout': 0.30,
            'use_imu_gyro_extrapolation': True,
            'max_imu_extrapolation': 0.12,
            'mode': 3,
            'timeout_sec': 0.35,
            'ramp_enabled': True,
            'accel_rpm_per_sec': 2.0,
            'decel_rpm_per_sec': 200.0,
        }
        
        # QoS profiles
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers and subscribers
        self.odom_pub = self.create_publisher(Odometry, 'odom', reliable_qos)
        self.runtime_config_pub = self.create_publisher(String, self.runtime_config_topic, reliable_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            best_effort_qos
        )
        self.max_linear_velocity_sub = self.create_subscription(
            Float32,
            self.max_linear_velocity_topic,
            self.max_linear_velocity_callback,
            reliable_qos
        )
        self.max_angular_velocity_sub = self.create_subscription(
            Float32,
            self.max_angular_velocity_topic,
            self.max_angular_velocity_callback,
            reliable_qos
        )
        self.max_rpm_sub = self.create_subscription(
            Float32,
            self.max_rpm_topic,
            self.max_rpm_callback,
            reliable_qos
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            self.emergency_stop_topic,
            self.emergency_stop_callback,
            reliable_qos
        )
        self.publish_rate_sub = self.create_subscription(
            Float32,
            self.publish_rate_topic,
            self.publish_rate_callback,
            reliable_qos
        )
        self.motor_command_timeout_sub = self.create_subscription(
            Float32,
            self.motor_command_timeout_topic,
            self.motor_command_timeout_callback,
            reliable_qos
        )
        self.balance_enabled_sub = self.create_subscription(
            Bool,
            self.balance_enabled_topic,
            self.balance_enabled_callback,
            reliable_qos
        )
        self.balance_kp_sub = self.create_subscription(
            Float32,
            self.balance_kp_topic,
            self.balance_kp_callback,
            reliable_qos
        )
        self.max_correction_rpm_sub = self.create_subscription(
            Float32,
            self.max_correction_rpm_topic,
            self.max_correction_rpm_callback,
            reliable_qos
        )
        self.safe_balance_ratio_sub = self.create_subscription(
            Float32,
            self.safe_balance_ratio_topic,
            self.safe_balance_ratio_callback,
            reliable_qos
        )
        self.use_imu_yaw_sub = self.create_subscription(
            Bool,
            self.use_imu_yaw_topic,
            self.use_imu_yaw_callback,
            reliable_qos
        )
        self.imu_yaw_offset_sub = self.create_subscription(
            Float32,
            self.imu_yaw_offset_topic,
            self.imu_yaw_offset_callback,
            reliable_qos
        )
        self.mode_sub = self.create_subscription(
            Float32,
            self.mode_topic,
            self.mode_callback,
            reliable_qos
        )
        self.timeout_sec_sub = self.create_subscription(
            Float32,
            self.timeout_sec_topic,
            self.timeout_sec_callback,
            reliable_qos
        )
        self.ramp_enabled_sub = self.create_subscription(
            Bool,
            self.ramp_enabled_topic,
            self.ramp_enabled_callback,
            reliable_qos
        )
        self.accel_rpm_per_sec_sub = self.create_subscription(
            Float32,
            self.accel_rpm_per_sec_topic,
            self.accel_rpm_per_sec_callback,
            reliable_qos
        )
        self.decel_rpm_per_sec_sub = self.create_subscription(
            Float32,
            self.decel_rpm_per_sec_topic,
            self.decel_rpm_per_sec_callback,
            reliable_qos
        )
        self.reset_runtime_config_sub = self.create_subscription(
            Bool,
            self.reset_runtime_config_topic,
            self.reset_runtime_config_callback,
            reliable_qos
        )

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            best_effort_qos
        )
        if self.use_imu_yaw:
            self.get_logger().info(f'Subscribed to IMU topic: {self.imu_topic}')

        # Timer for control loop
        self.timer = self.create_timer(1.0 / self.publish_rate, self.control_loop)
        self.runtime_config_timer = self.create_timer(
            1.0 / max(0.1, self.runtime_config_publish_rate),
            self.publish_runtime_config
        )
        
        # Balance control state
        self.balance_enabled = self.get_parameter('balance_enabled').value
        self.balance_kp = self.get_parameter('balance_kp').value
        self.max_correction_rpm = self.get_parameter('max_correction_rpm').value
        self.safe_balance_ratio = self.get_parameter('safe_balance_ratio').value
        self.mode = self.get_parameter('mode').value
        self.balance_ref_left = 0
        self.balance_ref_right = 0
        
        # For velocity calculation
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_theta = 0.0
        
        # Log startup info
        self.get_logger().info('=' * 60)
        self.get_logger().info('Differential Drive Controller Initialized')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Robot Parameters:')
        self.get_logger().info(f'  Wheel circumference: {self.wheel_circumference:.3f}m')
        self.get_logger().info(f'  Track width: {self.wheel_distance:.3f}m')
        self.get_logger().info(f'  Encoder ticks/rev: {self.ticks_per_rev}')
        self.get_logger().info(f'  Max linear velocity topic: {self.max_linear_velocity_topic}')
        self.get_logger().info(f'  Max angular velocity topic: {self.max_angular_velocity_topic}')
        self.get_logger().info(f'  Max RPM topic: {self.max_rpm_topic}')
        self.get_logger().info(f'  Emergency stop topic: {self.emergency_stop_topic}')
        self.get_logger().info(f'  Runtime config topic: {self.runtime_config_topic}')
        self.get_logger().info(f'  Reset runtime config topic: {self.reset_runtime_config_topic}')
        self.get_logger().info(f'')
        self.get_logger().info(f'Yaw Source: {"IMU" if self.use_imu_yaw else "Encoders"}')
        if self.use_imu_yaw:
            self.get_logger().info(f'  IMU Topic: {self.imu_topic}')
            self.get_logger().info(f'  Yaw Offset: {self.imu_yaw_offset:.3f} rad')
            self.get_logger().info(f'  Yaw Timeout: {self.imu_yaw_timeout:.3f}s')
            self.get_logger().info(f'  Gyro extrapolation: {self.use_imu_gyro_extrapolation}')
        self.get_logger().info(f'')
        self.get_logger().info(f'Physical Robot Control (cmd_vel mapping):')
        self.get_logger().info(f'  Swap wheels: {self.swap_cmd_vel_wheels}')
        self.get_logger().info(f'  Invert linear: {self.invert_cmd_vel_linear}')
        self.get_logger().info(f'  Invert angular: {self.invert_cmd_vel_angular}')
        self.get_logger().info(f'  Swap angular/linear: {self.swap_angular_linear_cmd_vel}')
        self.get_logger().info(f'')
        self.get_logger().info(f'Odometry Correction:')
        self.get_logger().info(f'  Swap wheels: {self.swap_odometry_wheels}')
        self.get_logger().info(f'  Invert linear: {self.invert_odometry_linear}')
        self.get_logger().info(f'  Invert angular: {self.invert_odometry_angular}')
        self.get_logger().info(f'  Invert left wheel: {self.invert_odometry_left}')
        self.get_logger().info(f'  Invert right wheel: {self.invert_odometry_right}')
        self.get_logger().info(f'  Swap angular/linear: {self.swap_angular_linear_odometry}')
        if self.swap_angular_linear_odometry:
            self.get_logger().warn(
                'swap_angular_linear_odometry is enabled. This can make rotation appear as linear odom drift.'
            )
        self.get_logger().info(f'')
        self.get_logger().info(f'Command Ramping:')
        self.get_logger().info(f'  Enabled: {self.ramp_enabled}')
        self.get_logger().info(f'  Accel: {self.accel_rpm_per_sec:.1f} RPM/s')
        self.get_logger().info(f'  Decel: {self.decel_rpm_per_sec:.1f} RPM/s')
        self.get_logger().info('=' * 60)

    @staticmethod
    def _slew_limit(current: int, target: int, accel_rate: float, decel_rate: float, dt: float) -> int:
        """Rate-limit changes from current->target, using accel/decel rates in RPM/s."""
        if dt <= 0.0 or current == target:
            return int(target)

        delta = target - current
        increasing_magnitude = abs(target) > abs(current)
        rate = accel_rate if increasing_magnitude else decel_rate
        max_step = max(0.0, rate) * dt
        if max_step <= 0.0:
            return int(current)

        if abs(delta) <= max_step:
            return int(target)
        step = int(max_step) if delta > 0 else -int(max_step)
        if step == 0:
            step = 1 if delta > 0 else -1
        return int(current + step)

    def get_ros_time_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def shortest_angle_delta(new_angle: float, old_angle: float) -> float:
        return math.atan2(math.sin(new_angle - old_angle), math.cos(new_angle - old_angle))
    
    def imu_callback(self, msg: Imu):
        """Callback for IMU data to get yaw angle."""
        try:
            # Extract quaternion from IMU message
            q = msg.orientation
            
            # Convert quaternion to yaw angle (Euler angle)
            # Yaw (z-axis rotation) = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Offset is now applied at the source in imu_publisher.py
            # to keep the whole system (including RViz/scans) synchronized.
            # yaw = yaw - self.imu_yaw_offset
            
            # Normalize to [-pi, pi]
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            if self.invert_imu_yaw:
                yaw = -yaw
            
            with self.imu_lock:
                self.imu_theta = yaw
                if self.invert_imu_yaw:
                    self.imu_angular_velocity_z = -float(msg.angular_velocity.z)
                else:
                    self.imu_angular_velocity_z = float(msg.angular_velocity.z)
                stamp_sec = self.stamp_to_sec(msg.header.stamp)
                self.imu_stamp_sec = stamp_sec if stamp_sec > 0.0 else self.get_ros_time_sec()
                self.imu_available = True
            
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')
    
    def get_fresh_imu_yaw(self) -> float | None:
        """Return recent IMU yaw, extrapolated slightly with gyro Z between IMU samples."""
        if not self.use_imu_yaw or not self.imu_available:
            return None

        with self.imu_lock:
            yaw = float(self.imu_theta)
            gyro_z = float(self.imu_angular_velocity_z)
            stamp_sec = self.imu_stamp_sec

        if stamp_sec is None:
            return yaw

        age = self.get_ros_time_sec() - stamp_sec
        if age < 0.0:
            age = 0.0
        if age > self.imu_yaw_timeout:
            self.get_logger().warn(
                f'IMU yaw is stale ({age:.3f}s old); falling back to encoder yaw.',
                throttle_duration_sec=2.0
            )
            return None

        if self.use_imu_gyro_extrapolation:
            yaw += gyro_z * min(age, max(0.0, self.max_imu_extrapolation))

        return self.normalize_angle(yaw)

    def get_current_yaw(self) -> float:
        """Get current yaw angle from fresh IMU data or encoder fallback."""
        imu_yaw = self.get_fresh_imu_yaw()
        if imu_yaw is not None:
            self.get_logger().info(f'YAW SOURCE: IMU | Yaw: {imu_yaw:.4f} rad', throttle_duration_sec=1.0)
            return imu_yaw
        else:
            self.get_logger().info(f'YAW SOURCE: ENCODER | Yaw: {self.theta:.4f} rad', throttle_duration_sec=1.0)
            return self.theta
    
    def update_odometry(self):
        """Update robot odometry from encoder readings with optional IMU yaw."""
        try:
            # Read current encoder ticks
            left_ticks_raw, right_ticks_raw = self.controller.read_encoders()
            
            # Apply odometry corrections
            if self.swap_odometry_wheels:
                # Swap left and right for odometry calculation
                left_ticks = right_ticks_raw
                right_ticks = left_ticks_raw
            else:
                left_ticks = left_ticks_raw
                right_ticks = right_ticks_raw

            # Time delta for velocity/RPM calculations
            current_time = self.get_ros_time_sec()
            dt = current_time - self.last_time
            
            # Calculate distance traveled by each wheel
            delta_left_ticks = left_ticks - self.last_left_ticks
            delta_right_ticks = right_ticks - self.last_right_ticks
            left_dist = self.ticks_to_distance(delta_left_ticks)
            right_dist = self.ticks_to_distance(delta_right_ticks)
            
            if self.invert_odometry_left:
                left_dist = -left_dist
            if self.invert_odometry_right:
                right_dist = -right_dist

            # Apply linear inversion if needed
            if self.invert_odometry_linear:
                left_dist = -left_dist
                right_dist = -right_dist
            
            # Calculate center distance (linear motion)
            if self.swap_angular_linear_odometry:
                # Legacy compatibility only. Prefer per-wheel odometry inversion instead.
                center_dist = (right_dist - left_dist)
            else:
                # Standard differential drive interpretation
                center_dist = (left_dist + right_dist) / 2.0

            encoder_delta_theta = (right_dist - left_dist) / self.wheel_distance
            if self.invert_odometry_angular:
                encoder_delta_theta = -encoder_delta_theta

            previous_theta = self.theta
            imu_yaw = self.get_fresh_imu_yaw()
            if imu_yaw is not None:
                new_theta = imu_yaw
                delta_theta = self.shortest_angle_delta(new_theta, previous_theta)
            else:
                delta_theta = encoder_delta_theta
                new_theta = self.normalize_angle(previous_theta + delta_theta)

            # Integrate translation at the middle heading. This reduces arc error while rotating.
            heading_mid = self.normalize_angle(previous_theta + 0.5 * delta_theta)
            delta_x = center_dist * math.cos(heading_mid)
            delta_y = center_dist * math.sin(heading_mid)
            
            self.x += delta_x
            self.y += delta_y
            self.theta = new_theta
            
            # Store for next iteration
            self.last_left_ticks = left_ticks
            self.last_right_ticks = right_ticks
            
            # Calculate velocities for odometry message
            if dt > 0 and hasattr(self, 'last_odom_x'):
                vx = center_dist / dt
                vz = delta_theta / dt
            else:
                vx = 0.0
                vz = 0.0

            # Calculate measured wheel RPM from encoder delta
            if dt > 0:
                l_rpm = (delta_left_ticks / self.ticks_per_rev) * (60.0 / dt)
                r_rpm = (delta_right_ticks / self.ticks_per_rev) * (60.0 / dt)
                if self.invert_odometry_left:
                    l_rpm = -l_rpm
                if self.invert_odometry_right:
                    r_rpm = -r_rpm
                if self.invert_odometry_linear:
                    l_rpm = -l_rpm
                    r_rpm = -r_rpm
                self.current_left_rpm = float(l_rpm)
                self.current_right_rpm = float(r_rpm)
            
            # Store for next iteration
            self.last_odom_x = self.x
            self.last_odom_y = self.y
            self.last_odom_theta = self.theta
            self.last_time = current_time
            
            return left_ticks_raw, right_ticks_raw, vx, vz
            
        except Exception as e:
            self.get_logger().error(f'Error in odometry update: {e}')
            return 0, 0, 0.0, 0.0
    
    def ticks_to_distance(self, ticks: int) -> float:
        """Convert encoder ticks to traveled distance."""
        return (ticks / self.ticks_per_rev) * self.wheel_circumference
    
    def rpm_to_linear_velocity(self, rpm: float) -> float:
        """Convert wheel RPM to linear velocity (m/s)."""
        return (rpm * self.wheel_circumference) / 60.0
    
    def linear_velocity_to_rpm(self, linear_vel: float) -> float:
        """Convert linear velocity (m/s) to wheel RPM."""
        return (linear_vel * 60.0) / self.wheel_circumference

    def get_runtime_config(self) -> dict:
        """Return the current runtime-tunable configuration."""
        return {
            'max_linear_velocity': float(self.max_linear_vel),
            'max_angular_velocity': float(self.max_angular_vel),
            'max_rpm': int(self.max_rpm),
            'publish_rate': float(self.publish_rate),
            'motor_command_timeout': float(self.motor_command_timeout),
            'balance_enabled': bool(self.balance_enabled),
            'balance_kp': float(self.balance_kp),
            'max_correction_rpm': int(self.max_correction_rpm),
            'safe_balance_ratio': float(self.safe_balance_ratio),
            'use_imu_yaw': bool(self.use_imu_yaw),
            'imu_yaw_offset': float(self.imu_yaw_offset),
            'imu_yaw_timeout': float(self.imu_yaw_timeout),
            'use_imu_gyro_extrapolation': bool(self.use_imu_gyro_extrapolation),
            'max_imu_extrapolation': float(self.max_imu_extrapolation),
            'mode': int(self.mode),
            'timeout_sec': float(self.get_parameter('timeout_sec').value),
            'ramp_enabled': bool(self.ramp_enabled),
            'accel_rpm_per_sec': float(self.accel_rpm_per_sec),
            'decel_rpm_per_sec': float(self.decel_rpm_per_sec),
            'emergency_stop_active': bool(self.emergency_stop_active),
        }

    def publish_runtime_config(self):
        """Publish the current runtime-tunable configuration as JSON."""
        msg = String()
        msg.data = json.dumps(self.get_runtime_config(), sort_keys=True)
        self.runtime_config_pub.publish(msg)

    def update_runtime_parameter(
        self,
        param_name: str,
        value,
        attr_name: str | None = None,
        param_type=None,
        validator=None,
        post_update=None,
    ) -> bool:
        """Update an in-memory field and ROS parameter with shared validation."""
        if validator is not None:
            error = validator(value)
            if error is not None:
                self.get_logger().warn(f'{param_name} update ignored: {error}')
                return False

        if attr_name is not None:
            setattr(self, attr_name, value)

        self.set_parameters([Parameter(param_name, param_type, value)])

        if post_update is not None:
            post_update()

        self.publish_runtime_config()
        self.get_logger().info(f'Updated {param_name}: {value}')
        return True

    @staticmethod
    def _positive_float_validator(value: float):
        if float(value) <= 0.0:
            return 'value must be positive'
        return None

    @staticmethod
    def _positive_int_validator(value: int):
        if int(value) <= 0:
            return 'value must be positive'
        return None

    @staticmethod
    def _non_zero_int_validator(value: int):
        if int(value) == 0:
            return 'value must be non-zero'
        return None

    @staticmethod
    def _non_negative_float_validator(value: float):
        if float(value) < 0.0:
            return 'value must be non-negative'
        return None

    @staticmethod
    def _ratio_validator(value: float):
        if float(value) < 0.0 or float(value) > 1.0:
            return 'value must be between 0.0 and 1.0'
        return None

    def _recreate_control_timer(self):
        self.timer.cancel()
        self.timer = self.create_timer(1.0 / self.publish_rate, self.control_loop)

    def _apply_motor_runtime_config(self):
        """Apply runtime motor settings that require writing to the motor controller."""
        self.target_left_rpm = 0
        self.target_right_rpm = 0
        self.sent_left_rpm = 0
        self.sent_right_rpm = 0
        self.last_motor_command_time = time.time()
        self.last_ramp_time = time.time()

        timeout_sec = float(self.get_parameter('timeout_sec').value)
        mode = int(self.get_parameter('mode').value)

        self.controller.stop()
        self.controller.left.node.timeout = timeout_sec
        self.controller.right.node.timeout = timeout_sec

        if self.get_parameter('nmt_start').value:
            self.io.send_nmt_start(0)
            time.sleep(0.15)

        self.controller.init(
            mode=mode,
            fix_din3=self.get_parameter('fix_din3').value
        )
        self.controller.stop()

        self.balance_ref_left, self.balance_ref_right = self.controller.read_encoders()
        self.last_left_ticks = self.balance_ref_left
        self.last_right_ticks = self.balance_ref_right
        self.get_logger().info(
            f'Applied motor runtime config: mode={mode}, timeout_sec={timeout_sec:.3f}'
        )

    def reset_runtime_config(self):
        """Restore runtime-tunable values to their defaults."""
        defaults = self.default_runtime_config

        self.max_linear_vel = float(defaults['max_linear_velocity'])
        self.max_angular_vel = float(defaults['max_angular_velocity'])
        self.max_rpm = int(defaults['max_rpm'])
        self.publish_rate = float(defaults['publish_rate'])
        self.motor_command_timeout = float(defaults['motor_command_timeout'])
        self.balance_enabled = bool(defaults['balance_enabled'])
        self.balance_kp = float(defaults['balance_kp'])
        self.max_correction_rpm = int(defaults['max_correction_rpm'])
        self.safe_balance_ratio = float(defaults['safe_balance_ratio'])
        self.use_imu_yaw = bool(defaults['use_imu_yaw'])
        self.imu_yaw_offset = float(defaults['imu_yaw_offset'])
        self.imu_yaw_timeout = float(defaults['imu_yaw_timeout'])
        self.use_imu_gyro_extrapolation = bool(defaults['use_imu_gyro_extrapolation'])
        self.max_imu_extrapolation = float(defaults['max_imu_extrapolation'])
        self.mode = int(defaults['mode'])
        self.ramp_enabled = bool(defaults['ramp_enabled'])
        self.accel_rpm_per_sec = float(defaults['accel_rpm_per_sec'])
        self.decel_rpm_per_sec = float(defaults['decel_rpm_per_sec'])
        self.emergency_stop_active = False
        self.target_left_rpm = 0
        self.target_right_rpm = 0
        self.sent_left_rpm = 0
        self.sent_right_rpm = 0
        self.last_motor_command_time = time.time()

        self.set_parameters([
            Parameter('max_linear_velocity', Parameter.Type.DOUBLE, self.max_linear_vel),
            Parameter('max_angular_velocity', Parameter.Type.DOUBLE, self.max_angular_vel),
            Parameter('max_rpm', Parameter.Type.INTEGER, self.max_rpm),
            Parameter('publish_rate', Parameter.Type.DOUBLE, self.publish_rate),
            Parameter('motor_command_timeout', Parameter.Type.DOUBLE, self.motor_command_timeout),
            Parameter('balance_enabled', Parameter.Type.BOOL, self.balance_enabled),
            Parameter('balance_kp', Parameter.Type.DOUBLE, self.balance_kp),
            Parameter('max_correction_rpm', Parameter.Type.INTEGER, self.max_correction_rpm),
            Parameter('safe_balance_ratio', Parameter.Type.DOUBLE, self.safe_balance_ratio),
            Parameter('use_imu_yaw', Parameter.Type.BOOL, self.use_imu_yaw),
            Parameter('imu_yaw_offset', Parameter.Type.DOUBLE, self.imu_yaw_offset),
            Parameter('imu_yaw_timeout', Parameter.Type.DOUBLE, self.imu_yaw_timeout),
            Parameter('use_imu_gyro_extrapolation', Parameter.Type.BOOL, self.use_imu_gyro_extrapolation),
            Parameter('max_imu_extrapolation', Parameter.Type.DOUBLE, self.max_imu_extrapolation),
            Parameter('mode', Parameter.Type.INTEGER, self.mode),
            Parameter('timeout_sec', Parameter.Type.DOUBLE, float(defaults['timeout_sec'])),
            Parameter('ramp_enabled', Parameter.Type.BOOL, self.ramp_enabled),
            Parameter('accel_rpm_per_sec', Parameter.Type.DOUBLE, self.accel_rpm_per_sec),
            Parameter('decel_rpm_per_sec', Parameter.Type.DOUBLE, self.decel_rpm_per_sec),
        ])

        self._recreate_control_timer()
        self.publish_runtime_config()
        try:
            self._apply_motor_runtime_config()
        except Exception as e:
            self.get_logger().error(f'Failed to apply reset runtime motor config live: {e}')
        else:
            self.get_logger().warn('Runtime config reset to defaults and applied to motors.')

    def reset_runtime_config_callback(self, msg: Bool):
        """Reset runtime-tunable values to defaults when a True message is received."""
        if not bool(msg.data):
            return
        self.reset_runtime_config()

    def max_linear_velocity_callback(self, msg: Float32):
        """Update max linear velocity in real time from a ROS2 topic."""
        self.update_runtime_parameter(
            'max_linear_velocity',
            float(msg.data),
            attr_name='max_linear_vel',
            param_type=Parameter.Type.DOUBLE,
            validator=self._positive_float_validator,
        )

    def max_angular_velocity_callback(self, msg: Float32):
        """Update max angular velocity in real time from a ROS2 topic."""
        self.update_runtime_parameter(
            'max_angular_velocity',
            float(msg.data),
            attr_name='max_angular_vel',
            param_type=Parameter.Type.DOUBLE,
            validator=self._positive_float_validator,
        )

    def max_rpm_callback(self, msg: Float32):
        """Update max RPM in real time from a ROS2 topic."""
        self.update_runtime_parameter(
            'max_rpm',
            int(msg.data),
            attr_name='max_rpm',
            param_type=Parameter.Type.INTEGER,
            validator=self._positive_int_validator,
        )

    def emergency_stop_callback(self, msg: Bool):
        """Latch/unlatch emergency stop from a ROS2 topic."""
        self.emergency_stop_active = bool(msg.data)

        if self.emergency_stop_active:
            self.target_left_rpm = 0
            self.target_right_rpm = 0
            self.sent_left_rpm = 0
            self.sent_right_rpm = 0
            self.last_motor_command_time = time.time()
            self.get_logger().warn('Emergency stop activated. Motor commands forced to zero.')
        else:
            self.last_motor_command_time = time.time()
            self.get_logger().info('Emergency stop cleared. Motion commands accepted again.')
        self.publish_runtime_config()

    def publish_rate_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'publish_rate',
            float(msg.data),
            attr_name='publish_rate',
            param_type=Parameter.Type.DOUBLE,
            validator=self._positive_float_validator,
            post_update=self._recreate_control_timer,
        )

    def motor_command_timeout_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'motor_command_timeout',
            float(msg.data),
            attr_name='motor_command_timeout',
            param_type=Parameter.Type.DOUBLE,
            validator=self._positive_float_validator,
        )

    def balance_enabled_callback(self, msg: Bool):
        self.update_runtime_parameter(
            'balance_enabled',
            bool(msg.data),
            attr_name='balance_enabled',
            param_type=Parameter.Type.BOOL,
        )

    def balance_kp_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'balance_kp',
            float(msg.data),
            attr_name='balance_kp',
            param_type=Parameter.Type.DOUBLE,
            validator=self._non_negative_float_validator,
        )

    def max_correction_rpm_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'max_correction_rpm',
            int(msg.data),
            attr_name='max_correction_rpm',
            param_type=Parameter.Type.INTEGER,
            validator=self._positive_int_validator,
        )

    def safe_balance_ratio_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'safe_balance_ratio',
            float(msg.data),
            attr_name='safe_balance_ratio',
            param_type=Parameter.Type.DOUBLE,
            validator=self._ratio_validator,
        )

    def use_imu_yaw_callback(self, msg: Bool):
        self.update_runtime_parameter(
            'use_imu_yaw',
            bool(msg.data),
            attr_name='use_imu_yaw',
            param_type=Parameter.Type.BOOL,
        )

    def imu_yaw_offset_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'imu_yaw_offset',
            float(msg.data),
            attr_name='imu_yaw_offset',
            param_type=Parameter.Type.DOUBLE,
        )

    def mode_callback(self, msg: Float32):
        if self.update_runtime_parameter(
            'mode',
            int(msg.data),
            attr_name='mode',
            param_type=Parameter.Type.INTEGER,
            validator=self._non_zero_int_validator,
        ):
            try:
                self._apply_motor_runtime_config()
            except Exception as e:
                self.get_logger().error(f'Failed to apply mode change live: {e}')

    def timeout_sec_callback(self, msg: Float32):
        if self.update_runtime_parameter(
            'timeout_sec',
            float(msg.data),
            param_type=Parameter.Type.DOUBLE,
            validator=self._positive_float_validator,
        ):
            try:
                self._apply_motor_runtime_config()
            except Exception as e:
                self.get_logger().error(f'Failed to apply timeout_sec change live: {e}')

    def ramp_enabled_callback(self, msg: Bool):
        self.update_runtime_parameter(
            'ramp_enabled',
            bool(msg.data),
            attr_name='ramp_enabled',
            param_type=Parameter.Type.BOOL,
        )

    def accel_rpm_per_sec_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'accel_rpm_per_sec',
            float(msg.data),
            attr_name='accel_rpm_per_sec',
            param_type=Parameter.Type.DOUBLE,
            validator=self._positive_float_validator,
        )

    def decel_rpm_per_sec_callback(self, msg: Float32):
        self.update_runtime_parameter(
            'decel_rpm_per_sec',
            float(msg.data),
            attr_name='decel_rpm_per_sec',
            param_type=Parameter.Type.DOUBLE,
            validator=self._positive_float_validator,
        )
    
    def cmd_vel_callback(self, msg: Twist):
        """Convert cmd_vel to wheel RPM commands with configurable mapping."""
        if self.emergency_stop_active:
            self.target_left_rpm = 0
            self.target_right_rpm = 0
            self.get_logger().warn(
                'Ignoring cmd_vel because emergency stop is active',
                throttle_duration_sec=1.0
            )
            return

        # Get linear and angular velocities
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z
        
        # Apply angular/linear swapping if needed
        if self.swap_angular_linear_cmd_vel:
            # Swap: linear becomes angular, angular becomes linear
            original_linear = linear_vel
            original_angular = angular_vel
            linear_vel = original_angular * (self.wheel_distance / 2.0)  # Convert angular to linear
            angular_vel = original_linear * 2.0 / self.wheel_distance   # Convert linear to angular
            self.get_logger().debug(f'Swapped cmd_vel: linear={linear_vel:.2f}, angular={angular_vel:.2f}')
        
        # Apply inversions
        if self.invert_cmd_vel_linear:
            linear_vel = -linear_vel
        
        if self.invert_cmd_vel_angular:
            angular_vel = -angular_vel
        
        # Clamp velocities to maximum values
        linear_vel = max(-self.max_linear_vel, min(self.max_linear_vel, linear_vel))
        angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, angular_vel))
        
        # Calculate wheel velocities using differential drive kinematics
        left_vel = linear_vel - (angular_vel * self.wheel_distance / 2.0)
        right_vel = linear_vel + (angular_vel * self.wheel_distance / 2.0)
        
        # Apply wheel swapping for cmd_vel if needed
        if self.swap_cmd_vel_wheels:
            left_vel, right_vel = right_vel, left_vel
        
        # Convert velocities to RPM
        left_rpm = self.linear_velocity_to_rpm(left_vel)
        right_rpm = self.linear_velocity_to_rpm(right_vel)
        
        # Clamp RPM to maximum values
        self.target_left_rpm = int(clamp(left_rpm, -self.max_rpm, self.max_rpm))
        self.target_right_rpm = int(clamp(right_rpm, -self.max_rpm, self.max_rpm))
        
        self.last_motor_command_time = time.time()
        
        # Log the command
        self.get_logger().info(
            f'cmd_vel: v={linear_vel:.2f}m/s, w={angular_vel:.2f}rad/s -> '
            f'Lcmd={self.target_left_rpm}RPM, Rcmd={self.target_right_rpm}RPM | '
            f'Lcur={self.current_left_rpm:.1f}RPM, Rcur={self.current_right_rpm:.1f}RPM'
        )
    



    def apply_balance_controll(self, target_left: int, target_right: int) -> tuple[int, int, int]:
        """Apply balance control to compensate for encoder differences."""
        if not self.balance_enabled:
            return target_left, target_right, 0
        
        try:
            # Read current encoder positions
            enc_left, enc_right = self.controller.read_encoders()
            
            # Calculate progress since reference
            progress_left = abs(enc_left - self.balance_ref_left)
            progress_right = abs(enc_right - self.balance_ref_right)
            error = progress_left - progress_right
            
            # Calculate dynamic correction limit based on speed
            avg_speed = (abs(target_left) + abs(target_right)) / 2.0
            dynamic_limit = int(max(5, avg_speed * max(0.0, min(1.0, self.safe_balance_ratio))))
            corr_limit = min(self.max_correction_rpm, dynamic_limit)
            
            # Apply correction
            correction = clamp(int(self.balance_kp * error), -corr_limit, corr_limit)
            
            # Apply correction to slower wheel
            if target_left * target_right >= 0:  # Same direction
                if error > 0:  # Left wheel is ahead
                    left_cmd = target_left
                    right_cmd = target_right + correction
                else:  # Right wheel is ahead
                    left_cmd = target_left + correction
                    right_cmd = target_right
            else:  # Different directions (turning)
                # For turning, adjust both wheels
                left_cmd = target_left - correction
                right_cmd = target_right + correction
            
            # Clamp commands
            left_cmd = clamp(left_cmd, -self.max_rpm, self.max_rpm)
            right_cmd = clamp(right_cmd, -self.max_rpm, self.max_rpm)
            
            return left_cmd, right_cmd, correction
        except Exception as e:
            self.get_logger().error(f'Error in balance control: {e}')
            return target_left, target_right, 0

    def apply_balance_control(self, target_left: int, target_right: int) -> tuple[int, int, int]:
        """Enhanced balance control: encoder + IMU heading correction."""

        if not self.balance_enabled:
            return target_left, target_right, 0

        try:
            # -----------------------------
            # 1. READ ENCODERS (position)
            # -----------------------------
            enc_left, enc_right = self.controller.read_encoders()

            progress_left = abs(enc_left - self.balance_ref_left)
            progress_right = abs(enc_right - self.balance_ref_right)
            encoder_error = progress_left - progress_right

            # -----------------------------
            # 2. HEADING CONTROL (IMU)
            # -----------------------------
            current_yaw = self.get_current_yaw()

            # Initialize target heading if not set
            if not hasattr(self, "target_yaw"):
                self.target_yaw = current_yaw

            yaw_error = self.target_yaw - current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

            # Tunable gain
            kp_heading = 0.5
            heading_correction = kp_heading * yaw_error

            # -----------------------------
            # 3. SPEED CONTROL (RPM feedback)
            # -----------------------------
            left_error = target_left - self.current_left_rpm
            right_error = target_right - self.current_right_rpm

            kp_speed = 0.2
            speed_corr_left = kp_speed * left_error
            speed_corr_right = kp_speed * right_error

            # -----------------------------
            # 4. ORIGINAL ENCODER BALANCE
            # -----------------------------
            avg_speed = (abs(target_left) + abs(target_right)) / 2.0
            dynamic_limit = int(max(5, avg_speed * max(0.0, min(1.0, self.safe_balance_ratio))))
            corr_limit = min(self.max_correction_rpm, dynamic_limit)

            encoder_correction = clamp(int(self.balance_kp * encoder_error), -corr_limit, corr_limit)

            # -----------------------------
            # 5. COMBINE ALL CONTROLS
            # -----------------------------
            left_cmd = target_left + speed_corr_left - heading_correction
            right_cmd = target_right + speed_corr_right + heading_correction

            # Apply encoder balancing on top
            if target_left * target_right >= 0:  # straight motion
                if encoder_error > 0:
                    right_cmd += encoder_correction
                else:
                    left_cmd += encoder_correction
            else:
                # rotation case
                left_cmd -= encoder_correction
                right_cmd += encoder_correction

            # -----------------------------
            # 6. CLAMP FINAL OUTPUT
            # -----------------------------
            left_cmd = clamp(int(left_cmd), -self.max_rpm, self.max_rpm)
            right_cmd = clamp(int(right_cmd), -self.max_rpm, self.max_rpm)

            return left_cmd, right_cmd, int(heading_correction)

        except Exception as e:
            self.get_logger().error(f'Error in balance control: {e}')
            return target_left, target_right, 0

    
    def publish_odometry(self, vx: float, vz: float):
        """Publish odometry and transform."""
        current_time = self.get_clock().now().to_msg()
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        
        # Set position
        odom_msg.pose.pose.position.x = float(self.x)
        odom_msg.pose.pose.position.y = float(self.y)
        odom_msg.pose.pose.position.z = 0.0
        
        # Set orientation (quaternion from yaw angle)
        q = self.euler_to_quaternion(0, 0, self.theta)
        odom_msg.pose.pose.orientation = q
        
        # Set velocity
        odom_msg.twist.twist.linear.x = float(vx)
        odom_msg.twist.twist.angular.z = float(vz)
        
        # Set covariance
        odom_msg.pose.covariance = self.odom_pose_covariance
        
        odom_msg.twist.covariance = self.odom_twist_covariance
        
        # Publish odometry
        self.odom_pub.publish(odom_msg)
        
        # Publish transform
        transform = TransformStamped()
        transform.header.stamp = current_time
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = float(self.x)
        transform.transform.translation.y = float(self.y)
        transform.transform.translation.z = 0.0
        transform.transform.rotation = q
        
        self.tf_broadcaster.sendTransform(transform)
    
    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles to quaternion."""
        q = Quaternion()
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        
        return q
    
    def control_loop(self):
        """Main control loop for odometry and motor control."""
        try:
            # Update odometry (now uses IMU yaw if available)
            left_ticks, right_ticks, vx, vz = self.update_odometry()

            if self.emergency_stop_active:
                self.target_left_rpm = 0
                self.target_right_rpm = 0
            
            # Check for motor command timeout
            if time.time() - self.last_motor_command_time > self.motor_command_timeout:
                if self.target_left_rpm != 0 or self.target_right_rpm != 0:
                    self.target_left_rpm = 0
                    self.target_right_rpm = 0
                    self.get_logger().debug('Motor command timeout, stopping motors')
            
            # Apply balance control
            cmd_left, cmd_right, correction = self.apply_balance_control(
                self.target_left_rpm, self.target_right_rpm
            )

            # Smooth acceleration/deceleration (slew-rate limit) before sending to motors
            now = time.time()
            dt_ramp = now - self.last_ramp_time
            self.last_ramp_time = now
            if self.ramp_enabled:
                cmd_left = self._slew_limit(
                    self.sent_left_rpm, cmd_left,
                    accel_rate=self.accel_rpm_per_sec,
                    decel_rate=self.decel_rpm_per_sec,
                    dt=dt_ramp
                )
                cmd_right = self._slew_limit(
                    self.sent_right_rpm, cmd_right,
                    accel_rate=self.accel_rpm_per_sec,
                    decel_rate=self.decel_rpm_per_sec,
                    dt=dt_ramp
                )

            self.sent_left_rpm = int(cmd_left)
            self.sent_right_rpm = int(cmd_right)
            
            # Send motor commands
            self.controller.set_pair(
                self.sent_left_rpm, self.sent_right_rpm,
                precise=self.get_parameter('precise_speed_command').value
            )
            
            # Publish odometry
            self.publish_odometry(vx, vz)
            
            # Update balance reference when stopped
            if self.balance_enabled and self.target_left_rpm == 0 and self.target_right_rpm == 0:
                self.balance_ref_left = left_ticks
                self.balance_ref_right = right_ticks
            
            # Log IMU status occasionally
            if not self.imu_available and self.use_imu_yaw:
                self.get_logger().warn('Waiting for IMU data...', throttle_duration_sec=5.0)
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
    
    def initialize_motors(self):
        """Initialize CANopen motors."""
        try:
            if self.get_parameter('nmt_start').value:
                self.io.send_nmt_start(0)
                time.sleep(0.15)
            
            self.controller.init(
                mode=self.get_parameter('mode').value,
                fix_din3=self.get_parameter('fix_din3').value
            )
            
            self.controller.stop()
            
            self.balance_ref_left, self.balance_ref_right = self.controller.read_encoders()
            self.last_left_ticks = self.balance_ref_left
            self.last_right_ticks = self.balance_ref_right

            # Reset ramp state to avoid jump on first command
            self.sent_left_rpm = 0
            self.sent_right_rpm = 0
            self.last_ramp_time = time.time()
            
            self.get_logger().info('Motors initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize motors: {e}')
            raise
    
    def shutdown_motors(self):
        """Shutdown motors safely."""
        try:
            self.controller.stop()
            if not self.get_parameter('keep_enabled').value:
                self.controller.disable()
            self.get_logger().info('Motors shutdown')
        except Exception as e:
            self.get_logger().error(f'Error during motor shutdown: {e}')
        finally:
            try:
                self.io.close()
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    
    node = DifferentialDriveController()
    
    try:
        node.initialize_motors()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    except Exception as e:
        node.get_logger().error(f'Error: {e}')
    finally:
        node.shutdown_motors()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
