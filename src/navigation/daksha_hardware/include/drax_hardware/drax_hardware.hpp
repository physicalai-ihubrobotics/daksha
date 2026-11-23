#ifndef DRAX_HARDWARE_HPP_
#define DRAX_HARDWARE_HPP_

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <optional>
#include <cstdint>

#include "drax_hardware/socket_can.hpp"
#include "drax_hardware/canopen_node.hpp"
#include "drax_hardware/motor.hpp"

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"

#include "rclcpp_lifecycle/state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/node.hpp"

#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_srvs/srv/set_bool.hpp"

#include "diagnostic_updater/diagnostic_updater.hpp"

namespace drax_hardware
{

class DraxHardware : public hardware_interface::SystemInterface
{
public:
    DraxHardware() = default;
    ~DraxHardware();

    //=========================================================
    // Lifecycle
    //=========================================================

    hardware_interface::CallbackReturn on_init(
        const hardware_interface::HardwareComponentInterfaceParams & params) override;

    hardware_interface::CallbackReturn on_configure(
        const rclcpp_lifecycle::State & previous_state) override;

    hardware_interface::CallbackReturn on_activate(
        const rclcpp_lifecycle::State & previous_state) override;

    hardware_interface::CallbackReturn on_deactivate(
        const rclcpp_lifecycle::State & previous_state) override;

    hardware_interface::CallbackReturn on_cleanup(
        const rclcpp_lifecycle::State & previous_state) override;

    hardware_interface::CallbackReturn on_shutdown(
        const rclcpp_lifecycle::State & previous_state) override;

    // Called by ros2_control when read()/write() return ERROR: stop the drives.
    hardware_interface::CallbackReturn on_error(
        const rclcpp_lifecycle::State & previous_state) override;

    //=========================================================
    // Interfaces
    //=========================================================

    std::vector<hardware_interface::StateInterface>
    export_state_interfaces() override;

    std::vector<hardware_interface::CommandInterface>
    export_command_interfaces() override;

    //=========================================================
    // Read / Write
    //=========================================================

    hardware_interface::return_type read(
        const rclcpp::Time & time,
        const rclcpp::Duration & period) override;

    hardware_interface::return_type write(
        const rclcpp::Time & time,
        const rclcpp::Duration & period) override;

private:
    //----------------------------------------------------------
    // Python algorithm helpers (exact ports)
    //----------------------------------------------------------

    // Python DifferentialDriveController._slew_limit()
    static int slewLimit(int current, int target,
                         double accel_rate, double decel_rate, double dt);

    // Python DifferentialDriveController.normalize_angle()
    static double normalizeAngle(double angle);

    // Python DifferentialDriveController.shortest_angle_delta()
    static double shortestAngleDelta(double new_angle, double old_angle);

    // Python DifferentialDriveController.ticks_to_distance()
    double ticksToDistance(int32_t ticks) const;

    // Python DifferentialDriveController.rpm_to_linear_velocity()
    double rpmToLinearVelocity(double rpm) const;

    // Python DifferentialDriveController.linear_velocity_to_rpm()
    double linearVelocityToRpm(double linear_vel) const;

    // Python DifferentialDriveController.get_fresh_imu_yaw()
    std::optional<double> getFreshImuYaw();

    // Python DifferentialDriveController.get_current_yaw()
    double getCurrentYaw();

    // Python DifferentialDriveController.apply_balance_control() (enhanced version)
    std::tuple<int, int, int> applyBalanceControl(int target_left, int target_right);

    // Read both encoders (left then right), returns false on failure
    bool readEncoders(int32_t & left, int32_t & right);

    // Python TwoMotorController.set_pair() — applies motor inversion, then writes
    bool motorSetPair(int left_rpm, int right_rpm, bool precise);

    // Python TwoMotorController.stop() = set_pair(0,0,precise=true)
    bool motorStop();

    // Python TwoMotorController.disable() = left.disable(); right.disable()
    void motorDisable();

    // Python DifferentialDriveController.initialize_motors()
    bool initializeMotors();

    // Python DifferentialDriveController.shutdown_motors()
    void shutdownMotors();

    // Python DifferentialDriveController._apply_motor_runtime_config()
    bool applyMotorRuntimeConfig();

    // Switch both CANopen nodes between init (long) and cyclic (short) SDO timeouts
    void setSdoPolicy(bool cyclic);

    hardware_interface::return_type readImpl(const rclcpp::Duration & period);
    hardware_interface::return_type writeImpl();

    // Best-effort zero-speed command; safe to call from any failure path
    void emergencyMotorStop(const char * reason);

    void closeBus();

    //----------------------------------------------------------
    // IMU callback (runs on internal node spin thread)
    //----------------------------------------------------------

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    //----------------------------------------------------------
    // Runtime parameter topic callbacks (Python subscribers)
    //----------------------------------------------------------

    void emergencyStopCallback(const std_msgs::msg::Bool::SharedPtr msg);

    //----------------------------------------------------------
    // Torque disable service callback
    //----------------------------------------------------------

    void handleSetTorqueDisable(
        const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
        std::shared_ptr<std_srvs::srv::SetBool::Response> res);

    //----------------------------------------------------------
    // Diagnostics
    //----------------------------------------------------------

    void updateDiagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat);
    void addMotorDiagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat,
                             const std::string & name, Motor * motor);

    //----------------------------------------------------------
    // Internal spin thread (IMU + runtime param subscriptions)
    //----------------------------------------------------------

    void startInternalNode();
    void stopInternalNode();

    //==========================================================
    // Data members
    //==========================================================

    hardware_interface::HardwareInfo info_;

    //----------------------------------------------------------
    // CAN configuration
    //----------------------------------------------------------

    std::string can_interface_;
    uint8_t left_node_id_  = 1;
    uint8_t right_node_id_ = 2;

    // Python: timeout_sec (0.35), retries (3)
    std::chrono::milliseconds sdo_timeout_{350};
    int sdo_retries_ = 3;

    // SDO policy inside read()/write(). Measured round trip on can0: median 0.4 ms, max 7 ms.
    std::chrono::milliseconds cyclic_sdo_timeout_{30};
    int cyclic_sdo_retries_ = 2;

    // Serialises all CAN transactions: control loop vs. torque service thread.
    // SDO request/response on one socket must never interleave.
    std::mutex can_mutex_;

    //----------------------------------------------------------
    // CAN objects
    //----------------------------------------------------------

    std::unique_ptr<SocketCanBus>  bus_;
    std::unique_ptr<CanOpenNode>   left_node_;
    std::unique_ptr<CanOpenNode>   right_node_;
    std::unique_ptr<Motor>         left_motor_;
    std::unique_ptr<Motor>         right_motor_;
    MotorConfig motor_config_;

    //----------------------------------------------------------
    // Robot geometry (Python constructor params)
    //----------------------------------------------------------

    double wheel_circumference_ = 0.575;  // meters
    double wheel_distance_      = 0.50;   // track width, meters
    double ticks_per_rev_       = 4096.0; // encoder ticks per revolution

    //----------------------------------------------------------
    // Motor inversion (Python TwoMotorController)
    //----------------------------------------------------------

    bool invert_left_  = false;
    bool invert_right_ = true;

    //----------------------------------------------------------
    // CANopen init params (Python constructor)
    //----------------------------------------------------------

    bool    nmt_start_              = true;
    bool    fix_din3_               = true;
    int8_t  motor_mode_             = -3;
    double  timeout_sec_            = 0.35;
    int     retries_                = 3;
    bool    precise_speed_command_  = true;
    bool    keep_enabled_           = false;

    //----------------------------------------------------------
    // Runtime-tunable parameters (Python runtime params)
    //----------------------------------------------------------

    double max_linear_vel_        = 1.0;
    double max_angular_vel_       = 0.8;
    int    max_rpm_               = 30;
    double motor_command_timeout_ = 0.5;   // seconds

    //----------------------------------------------------------
    // Odometry correction parameters
    //----------------------------------------------------------

    bool swap_odometry_wheels_       = true;
    bool invert_odometry_linear_     = false;
    bool invert_odometry_angular_    = false;
    bool invert_odometry_left_       = true;
    bool invert_odometry_right_      = false;
    bool swap_angular_linear_odometry_ = false;

    //----------------------------------------------------------
    // CMD_VEL mapping corrections (Python cmd_vel_callback)
    //----------------------------------------------------------

    bool swap_cmd_vel_wheels_           = false;
    bool invert_cmd_vel_angular_        = true;   // Python default: True
    bool invert_cmd_vel_linear_         = false;
    bool swap_angular_linear_cmd_vel_   = false;

    //----------------------------------------------------------
    // Odometry state
    //----------------------------------------------------------

    double  x_     = 0.0;
    double  y_     = 0.0;
    double  theta_ = 0.0;  // current heading (radians)

    int32_t last_left_ticks_   = 0;
    int32_t last_right_ticks_  = 0;
    int32_t balance_ref_left_  = 0;
    int32_t balance_ref_right_ = 0;

    double current_left_rpm_  = 0.0;  // measured from encoders
    double current_right_rpm_ = 0.0;

    rclcpp::Time last_odom_time_;
    bool         first_odom_ = true;

    //----------------------------------------------------------
    // ros2_control state/command interfaces
    //----------------------------------------------------------

    double left_wheel_position_  = 0.0;  // rad
    double right_wheel_position_ = 0.0;
    double left_wheel_velocity_  = 0.0;  // rad/s
    double right_wheel_velocity_ = 0.0;
    double left_wheel_command_   = 0.0;  // rad/s (from diff_drive_controller)
    double right_wheel_command_  = 0.0;

    //----------------------------------------------------------
    // Motor command / safety
    //----------------------------------------------------------

    int  target_left_rpm_   = 0;
    int  target_right_rpm_  = 0;
    int  sent_left_rpm_     = 0;
    int  sent_right_rpm_    = 0;

    std::chrono::steady_clock::time_point last_motor_command_time_;
    std::atomic<bool> emergency_stop_active_{false};
    std::atomic<bool> torque_disabled_{false};

    //----------------------------------------------------------
    // Slew rate limiter (Python ramp)
    //----------------------------------------------------------

    bool   ramp_enabled_       = true;
    double accel_rpm_per_sec_  = 0.05;
    double decel_rpm_per_sec_  = 200.0;

    std::chrono::steady_clock::time_point last_ramp_time_;

    //----------------------------------------------------------
    // Balance control (Python enhanced apply_balance_control)
    //----------------------------------------------------------

    bool   balance_enabled_      = false;
    double balance_kp_           = 0.002;
    int    max_correction_rpm_   = 20;
    double safe_balance_ratio_   = 0.15;

    double target_yaw_           = 0.0;
    bool   target_yaw_initialized_ = false;

    //----------------------------------------------------------
    // IMU (Python imu_callback / get_fresh_imu_yaw)
    //----------------------------------------------------------

    bool   use_imu_yaw_               = true;
    std::string imu_topic_            = "/bno055/imu";
    double imu_yaw_timeout_           = 0.30;
    bool   use_imu_gyro_extrapolation_= true;
    double max_imu_extrapolation_     = 0.12;
    bool   invert_imu_yaw_            = false;

    // Protected by imu_mutex_
    mutable std::mutex imu_mutex_;
    double imu_theta_              = 0.0;
    double imu_angular_velocity_z_ = 0.0;
    double imu_stamp_sec_          = 0.0;
    bool   imu_available_          = false;

    //----------------------------------------------------------
    // Emergency stop topic
    //----------------------------------------------------------

    std::string emergency_stop_topic_ = "/emergency_stop";

    //----------------------------------------------------------
    // Internal ROS node for subscriptions (IMU, e-stop, etc.)
    //----------------------------------------------------------

    rclcpp::Node::SharedPtr internal_node_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr internal_executor_;
    std::thread internal_spin_thread_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr   emergency_stop_sub_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr torque_disable_srv_;

    //----------------------------------------------------------
    // Diagnostics
    //----------------------------------------------------------

    rclcpp::Node::SharedPtr diagnostics_node_;
    std::unique_ptr<diagnostic_updater::Updater> diagnostics_updater_;
    rclcpp::Time last_diagnostics_update_;
    double       diagnostics_period_sec_ = 1.0;

    //----------------------------------------------------------
    // Encoder read failure tracking
    //----------------------------------------------------------

    static constexpr int MAX_CONSECUTIVE_READ_FAILURES = 10;
    static constexpr int MAX_CONSECUTIVE_WRITE_FAILURES = 10;
    int left_consecutive_read_failures_  = 0;
    int right_consecutive_read_failures_ = 0;
    int write_consecutive_failures_      = 0;

    // raw encoder values for balance ref
    int32_t left_encoder_  = 0;
    int32_t right_encoder_ = 0;
};

} // namespace drax_hardware

#endif // DRAX_HARDWARE_HPP_
