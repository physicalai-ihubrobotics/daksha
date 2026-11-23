/**
 * drax_hardware.cpp
 *
 * C++ ros2_control hardware interface for the Drax differential drive robot.
 *
 * This file is a faithful port of the Python DifferentialDriveController
 * (motor.py) into the hardware_interface::SystemInterface API.
 *
 * Every algorithm, formula, constant, sequence, and timeout is derived
 * directly from motor.py / cmd_vel.py and is preserved exactly.
 *
 * Python reference:
 *   /home/fayas/Desktop/amr_ws/src/drax_hardware/tools/motor.py
 *   /home/fayas/Desktop/amr_ws/src/drax_hardware/tools/cmd_vel.py
 */

#include "drax_hardware/drax_hardware.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <optional>
#include <tuple>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

namespace drax_hardware
{

// ============================================================
// Utility helpers (file-local)
// ============================================================
namespace
{

rclcpp::Logger logger()
{
    return rclcpp::get_logger("DraxHardware");
}

// Read a string parameter from URDF hardware_parameters with a default.
std::string paramOr(const hardware_interface::HardwareInfo & info,
                    const std::string & name,
                    const std::string & default_value)
{
    const auto it = info.hardware_parameters.find(name);
    return it == info.hardware_parameters.end() ? default_value : it->second;
}

int intParamOr(const hardware_interface::HardwareInfo & info,
               const std::string & name, int default_value)
{
    return std::stoi(paramOr(info, name, std::to_string(default_value)));
}

double doubleParamOr(const hardware_interface::HardwareInfo & info,
                     const std::string & name, double default_value)
{
    return std::stod(paramOr(info, name, std::to_string(default_value)));
}

bool boolParamOr(const hardware_interface::HardwareInfo & info,
                 const std::string & name, bool default_value)
{
    auto v = paramOr(info, name, default_value ? "true" : "false");
    // Convert to lowercase for case-insensitive comparison
    // (xacro outputs "True"/"False" with capital first letter)
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return v == "true" || v == "1" || v == "yes" || v == "on";
}

uint16_t indexParamOr(const hardware_interface::HardwareInfo & info,
                      const std::string & name, uint16_t default_value)
{
    return static_cast<uint16_t>(
        std::stoul(paramOr(info, name, std::to_string(default_value)), nullptr, 0));
}

uint8_t subindexParamOr(const hardware_interface::HardwareInfo & info,
                        const std::string & name, uint8_t default_value)
{
    return static_cast<uint8_t>(intParamOr(info, name, default_value));
}

// Python: clamp(value, low, high)
template<typename T>
T clamp(T value, T low, T high)
{
    return std::max(low, std::min(high, value));
}

} // namespace

// ============================================================
// Destructor
// ============================================================
DraxHardware::~DraxHardware()
{
    // Covers Ctrl-C / process teardown where on_deactivate is never called:
    // never leave the drives running at the last commanded RPM.
    stopInternalNode();
    if (bus_ && bus_->isOpen())
    {
        emergencyMotorStop("hardware interface destroyed");
    }
    closeBus();
}

// ============================================================
// on_init — Read all parameters from URDF hardware_parameters
//   Mirrors Python DifferentialDriveController.__init__() parameter reading.
// ============================================================
hardware_interface::CallbackReturn
DraxHardware::on_init(const hardware_interface::HardwareComponentInterfaceParams & params)
{
    const auto & info = params.hardware_info;
    if (hardware_interface::SystemInterface::on_init(params) !=
        hardware_interface::CallbackReturn::SUCCESS)
    {
        return hardware_interface::CallbackReturn::ERROR;
    }

    info_ = info;

    // ---- Joint layout: exactly [left, right], each with a velocity command ----
    if (info_.joints.size() != 2)
    {
        RCLCPP_FATAL(logger(), "Expected 2 joints (left, right) in <ros2_control>, got %zu",
                     info_.joints.size());
        return hardware_interface::CallbackReturn::ERROR;
    }
    for (const auto & joint : info_.joints)
    {
        if (joint.command_interfaces.size() != 1 ||
            joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY)
        {
            RCLCPP_FATAL(logger(), "Joint '%s' must have exactly one 'velocity' command interface",
                         joint.name.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }
    }

    // std::stoi/stod throw on malformed params (e.g. an unexpanded xacro property);
    // an exception escaping on_init would take down ros2_control_node.
    try
    {
        // ---- CAN / node IDs ----
        can_interface_  = paramOr(info_, "can_interface", "can0");
        left_node_id_   = static_cast<uint8_t>(intParamOr(info_, "left_node_id",  1));
        right_node_id_  = static_cast<uint8_t>(intParamOr(info_, "right_node_id", 2));

        // ---- Robot geometry (Python defaults) ----
        wheel_circumference_ = doubleParamOr(info_, "wheel_circumference", 0.575);
        wheel_distance_      = doubleParamOr(info_, "wheel_distance",      0.50);
        ticks_per_rev_       = doubleParamOr(info_, "ticks_per_revolution", 4096.0);

        // ---- Motor inversion (Python TwoMotorController) ----
        invert_left_  = boolParamOr(info_, "invert_left",  false);
        invert_right_ = boolParamOr(info_, "invert_right", true);

        // ---- CANopen init params ----
        nmt_start_             = boolParamOr(info_, "nmt_start",  true);
        fix_din3_              = boolParamOr(info_, "fix_din3",   true);
        motor_mode_            = static_cast<int8_t>(intParamOr(info_, "mode", -3));
        timeout_sec_           = doubleParamOr(info_, "timeout_sec", 0.35);
        retries_               = intParamOr(info_,   "retries",     3);
        precise_speed_command_ = boolParamOr(info_,  "precise_speed_command", true);
        keep_enabled_          = boolParamOr(info_,  "keep_enabled", false);

        sdo_timeout_  = std::chrono::milliseconds(
                            static_cast<int>(timeout_sec_ * 1000.0));
        sdo_retries_  = std::max(1, retries_);

        cyclic_sdo_timeout_ = std::chrono::milliseconds(static_cast<int>(
            doubleParamOr(info_, "cyclic_timeout_sec", 0.03) * 1000.0));
        cyclic_sdo_retries_ = std::max(1, intParamOr(info_, "cyclic_retries", 2));

        // ---- Diagnostics ----
        motor_config_.fix_din3 = fix_din3_;
        motor_config_.feedback_resolution = intParamOr(info_, "feedback_resolution", 4096);
        motor_config_.temperature_index   = indexParamOr(info_, "diagnostic_temperature_index",   0);
        motor_config_.temperature_subindex = subindexParamOr(info_, "diagnostic_temperature_subindex", 0);
        motor_config_.temperature_scale   = doubleParamOr(info_, "diagnostic_temperature_scale", 1.0);
        motor_config_.voltage_index       = indexParamOr(info_, "diagnostic_voltage_index",   0);
        motor_config_.voltage_subindex    = subindexParamOr(info_, "diagnostic_voltage_subindex", 0);
        motor_config_.voltage_scale       = doubleParamOr(info_, "diagnostic_voltage_scale", 1.0);
        diagnostics_period_sec_           = doubleParamOr(info_, "diagnostics_period_sec", 1.0);

        // ---- Runtime-tunable parameters (Python defaults) ----
        max_linear_vel_        = doubleParamOr(info_, "max_linear_velocity",    1.0);
        max_angular_vel_       = doubleParamOr(info_, "max_angular_velocity",   0.8);
        max_rpm_               = intParamOr(info_,    "max_rpm",                30);
        motor_command_timeout_ = doubleParamOr(info_, "motor_command_timeout",  0.5);

        // ---- Odometry correction (Python defaults) ----
        swap_odometry_wheels_         = boolParamOr(info_, "swap_odometry_wheels",         true);
        invert_odometry_linear_       = boolParamOr(info_, "invert_odometry_linear",       false);
        invert_odometry_angular_      = boolParamOr(info_, "invert_odometry_angular",      false);
        invert_odometry_left_         = boolParamOr(info_, "invert_odometry_left",         true);
        invert_odometry_right_        = boolParamOr(info_, "invert_odometry_right",        false);
        swap_angular_linear_odometry_ = boolParamOr(info_, "swap_angular_linear_odometry", false);

        // ---- CMD_VEL mapping corrections (Python defaults) ----
        swap_cmd_vel_wheels_         = boolParamOr(info_, "swap_cmd_vel_wheels",         false);
        invert_cmd_vel_angular_      = boolParamOr(info_, "invert_cmd_vel_angular",      true);   // Python default: True
        invert_cmd_vel_linear_       = boolParamOr(info_, "invert_cmd_vel_linear",       false);
        swap_angular_linear_cmd_vel_ = boolParamOr(info_, "swap_angular_linear_cmd_vel", false);

        // ---- Balance control ----
        balance_enabled_    = boolParamOr(info_,   "balance_enabled",    false);
        balance_kp_         = doubleParamOr(info_, "balance_kp",         0.002);
        max_correction_rpm_ = intParamOr(info_,    "max_correction_rpm", 20);
        safe_balance_ratio_ = doubleParamOr(info_, "safe_balance_ratio", 0.15);

        // ---- IMU ----
        use_imu_yaw_                = boolParamOr(info_,   "use_imu_yaw",                true);
        imu_topic_                  = paramOr(info_,       "imu_topic",                  "/bno055/imu");
        imu_yaw_timeout_            = doubleParamOr(info_, "imu_yaw_timeout",            0.30);
        use_imu_gyro_extrapolation_ = boolParamOr(info_,   "use_imu_gyro_extrapolation", true);
        max_imu_extrapolation_      = doubleParamOr(info_, "max_imu_extrapolation",      0.12);
        invert_imu_yaw_             = boolParamOr(info_,   "invert_imu_yaw",             false);

        // ---- Slew rate limiter (Python ramp) ----
        ramp_enabled_      = boolParamOr(info_,   "ramp_enabled",      true);
        accel_rpm_per_sec_ = doubleParamOr(info_, "accel_rpm_per_sec", 0.05);
        decel_rpm_per_sec_ = doubleParamOr(info_, "decel_rpm_per_sec", 200.0);

        // ---- Safety topics ----
        emergency_stop_topic_ = paramOr(info_, "emergency_stop_topic", "/emergency_stop");
    }
    catch (const std::exception & e)
    {
        RCLCPP_FATAL(logger(), "Invalid <hardware> parameter in URDF: %s", e.what());
        return hardware_interface::CallbackReturn::ERROR;
    }

    // Values used as divisors / limits must be sane, or odometry and RPM go NaN/inf.
    if (!(wheel_circumference_ > 0.0) || !(wheel_distance_ > 0.0) || !(ticks_per_rev_ > 0.0) ||
        max_rpm_ <= 0 || sdo_timeout_.count() <= 0 || cyclic_sdo_timeout_.count() <= 0)
    {
        RCLCPP_FATAL(logger(),
                     "Invalid params: wheel_circumference=%.4f wheel_distance=%.4f "
                     "ticks_per_revolution=%.1f max_rpm=%d timeout_sec=%.3f cyclic_timeout_sec=%.3f "
                     "(all must be > 0)",
                     wheel_circumference_, wheel_distance_, ticks_per_rev_, max_rpm_,
                     timeout_sec_, cyclic_sdo_timeout_.count() / 1000.0);
        return hardware_interface::CallbackReturn::ERROR;
    }

    RCLCPP_INFO(logger(), "============================================================");
    RCLCPP_INFO(logger(), "DraxHardware: on_init()");
    RCLCPP_INFO(logger(), "  CAN interface: %s", can_interface_.c_str());
    RCLCPP_INFO(logger(), "  Left node ID: %u  (invert: %s)",
                left_node_id_,  invert_left_  ? "yes" : "no");
    RCLCPP_INFO(logger(), "  Right node ID: %u  (invert: %s)",
                right_node_id_, invert_right_ ? "yes" : "no");
    RCLCPP_INFO(logger(), "  Wheel circumference: %.3f m", wheel_circumference_);
    RCLCPP_INFO(logger(), "  Track width: %.3f m",        wheel_distance_);
    RCLCPP_INFO(logger(), "  Ticks/rev: %.0f",            ticks_per_rev_);
    RCLCPP_INFO(logger(), "  Mode: %d  timeout_sec: %.3f  retries: %d  (cyclic: %ld ms x %d)",
                motor_mode_, timeout_sec_, retries_,
                static_cast<long>(cyclic_sdo_timeout_.count()), cyclic_sdo_retries_);
    RCLCPP_INFO(logger(), "  Precise speed command: %s",
                precise_speed_command_ ? "yes" : "no");
    RCLCPP_INFO(logger(), "  Yaw source: %s",
                use_imu_yaw_ ? "IMU" : "Encoders");
    if (use_imu_yaw_) {
        RCLCPP_INFO(logger(), "  IMU topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(logger(), "  IMU timeout: %.3f s  gyro extrap: %s  max extrap: %.3f",
                    imu_yaw_timeout_,
                    use_imu_gyro_extrapolation_ ? "yes" : "no",
                    max_imu_extrapolation_);
    }
    RCLCPP_INFO(logger(), "  Odometry corrections: swap_wheels=%s  inv_linear=%s"
                          "  inv_angular=%s  inv_left=%s  inv_right=%s",
                swap_odometry_wheels_    ? "y" : "n",
                invert_odometry_linear_  ? "y" : "n",
                invert_odometry_angular_ ? "y" : "n",
                invert_odometry_left_    ? "y" : "n",
                invert_odometry_right_   ? "y" : "n");
    RCLCPP_INFO(logger(), "  CMD_VEL mapping: swap_wheels=%s  inv_angular=%s"
                          "  inv_linear=%s  swap_ang_lin=%s",
                swap_cmd_vel_wheels_         ? "y" : "n",
                invert_cmd_vel_angular_      ? "y" : "n",
                invert_cmd_vel_linear_       ? "y" : "n",
                swap_angular_linear_cmd_vel_ ? "y" : "n");
    RCLCPP_INFO(logger(), "  Balance: enabled=%s  kp=%.4f  max_corr=%d  ratio=%.3f",
                balance_enabled_ ? "yes" : "no",
                balance_kp_, max_correction_rpm_, safe_balance_ratio_);
    RCLCPP_INFO(logger(), "  Ramp: enabled=%s  accel=%.2f  decel=%.2f RPM/s",
                ramp_enabled_ ? "yes" : "no",
                accel_rpm_per_sec_, decel_rpm_per_sec_);
    RCLCPP_INFO(logger(), "============================================================");

    return hardware_interface::CallbackReturn::SUCCESS;
}

// ============================================================
// on_configure — Open CAN bus, create nodes and motors.
//   Mirrors Python __init__ CAN/motor construction.
// ============================================================
hardware_interface::CallbackReturn
DraxHardware::on_configure(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(logger(), "Opening CAN interface: %s", can_interface_.c_str());

    bus_ = std::make_unique<SocketCanBus>(logger());
    if (!bus_->open(can_interface_))
    {
        RCLCPP_ERROR(logger(), "Failed to open CAN interface %s", can_interface_.c_str());
        return hardware_interface::CallbackReturn::ERROR;
    }

    left_node_ = std::make_unique<CanOpenNode>(
        *bus_, left_node_id_,
        rclcpp::get_logger("DraxHardware.LeftNode"),
        sdo_timeout_, sdo_retries_);

    right_node_ = std::make_unique<CanOpenNode>(
        *bus_, right_node_id_,
        rclcpp::get_logger("DraxHardware.RightNode"),
        sdo_timeout_, sdo_retries_);

    left_motor_ = std::make_unique<Motor>(
        *left_node_, motor_config_,
        rclcpp::get_logger("DraxHardware.LeftMotor"));

    right_motor_ = std::make_unique<Motor>(
        *right_node_, motor_config_,
        rclcpp::get_logger("DraxHardware.RightMotor"));

    // Diagnostics
    diagnostics_node_    = rclcpp::Node::make_shared("drax_hardware_diagnostics");
    diagnostics_updater_ = std::make_unique<diagnostic_updater::Updater>(diagnostics_node_);
    diagnostics_updater_->setHardwareID("drax_hardware");
    diagnostics_updater_->add("motor_status", this, &DraxHardware::updateDiagnostics);
    last_diagnostics_update_ = diagnostics_node_->now();

    RCLCPP_INFO(logger(), "CAN bus open, motors created.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

// ============================================================
// on_activate — Motor initialisation sequence.
//   Mirrors Python DifferentialDriveController.initialize_motors().
// ============================================================
hardware_interface::CallbackReturn
DraxHardware::on_activate(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(logger(), "Activating DraxHardware...");

    // Reset state
    x_     = 0.0;
    y_     = 0.0;
    theta_ = 0.0;
    left_wheel_position_  = 0.0;
    right_wheel_position_ = 0.0;
    left_wheel_velocity_  = 0.0;
    right_wheel_velocity_ = 0.0;
    left_wheel_command_   = 0.0;
    right_wheel_command_  = 0.0;
    target_left_rpm_      = 0;
    target_right_rpm_     = 0;
    sent_left_rpm_        = 0;
    sent_right_rpm_       = 0;
    current_left_rpm_     = 0.0;
    current_right_rpm_    = 0.0;
    emergency_stop_active_.store(false);
    target_yaw_initialized_ = false;
    first_odom_ = true;
    left_consecutive_read_failures_  = 0;
    right_consecutive_read_failures_ = 0;
    write_consecutive_failures_      = 0;

    const auto now = std::chrono::steady_clock::now();
    last_motor_command_time_ = now;
    last_ramp_time_          = now;

    {
        std::lock_guard<std::mutex> lock(can_mutex_);

        // Re-activation after on_deactivate/on_error: the socket may have been closed
        // or the interface reset (bus-off, cable).
        if (!bus_->isOpen() && !bus_->open(can_interface_))
        {
            RCLCPP_ERROR(logger(), "Failed to reopen CAN interface %s", can_interface_.c_str());
            return hardware_interface::CallbackReturn::ERROR;
        }

        // Python initialize_motors() sequence
        setSdoPolicy(false);
        const bool ok = initializeMotors();
        setSdoPolicy(true);
        if (!ok)
        {
            emergencyMotorStop("activation failed");
            return hardware_interface::CallbackReturn::ERROR;
        }
    }

    // Spin up internal node for IMU + e-stop subscriptions
    startInternalNode();

    RCLCPP_INFO(logger(), "DraxHardware activated.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

// ============================================================
// on_deactivate — Python shutdown_motors()
// ============================================================
hardware_interface::CallbackReturn
DraxHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(logger(), "Deactivating DraxHardware...");

    torque_disabled_.store(false);

    stopInternalNode();
    {
        std::lock_guard<std::mutex> lock(can_mutex_);
        setSdoPolicy(false);
        shutdownMotors();
    }

    // CAN socket and diagnostics stay open so the component can be re-activated;
    // they are released in on_cleanup / on_shutdown.
    RCLCPP_INFO(logger(), "DraxHardware deactivated.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
DraxHardware::on_cleanup(const rclcpp_lifecycle::State &)
{
    stopInternalNode();
    closeBus();
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
DraxHardware::on_shutdown(const rclcpp_lifecycle::State &)
{
    stopInternalNode();
    if (bus_ && bus_->isOpen())
    {
        emergencyMotorStop("shutdown");
    }
    closeBus();
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
DraxHardware::on_error(const rclcpp_lifecycle::State &)
{
    RCLCPP_ERROR(logger(), "Hardware error: stopping motors");
    stopInternalNode();
    torque_disabled_.store(false);
    if (bus_ && bus_->isOpen())
    {
        emergencyMotorStop("hardware error");
    }
    // SUCCESS -> component goes to UNCONFIGURED and can be configured/activated again.
    closeBus();
    return hardware_interface::CallbackReturn::SUCCESS;
}

void DraxHardware::closeBus()
{
    std::lock_guard<std::mutex> lock(can_mutex_);
    if (bus_)
    {
        bus_->close();
    }
    diagnostics_updater_.reset();
    diagnostics_node_.reset();
}

void DraxHardware::setSdoPolicy(bool cyclic)
{
    const auto timeout = cyclic ? cyclic_sdo_timeout_ : sdo_timeout_;
    const int  retries = cyclic ? cyclic_sdo_retries_ : sdo_retries_;
    if (left_node_)  { left_node_->setRequestPolicy(timeout, retries); }
    if (right_node_) { right_node_->setRequestPolicy(timeout, retries); }
}

// Caller must NOT hold can_mutex_. Uses short timeouts: this runs on failure paths
// where the drive may be unreachable, and must not block for seconds.
void DraxHardware::emergencyMotorStop(const char * reason)
{
    std::lock_guard<std::mutex> lock(can_mutex_);
    if (!bus_ || !bus_->isOpen() || !left_motor_ || !right_motor_)
    {
        return;
    }
    RCLCPP_WARN(logger(), "Stopping motors (%s)", reason);
    setSdoPolicy(true);
    try
    {
        const bool l_ok = left_motor_->stop();
        const bool r_ok = right_motor_->stop();
        if (!keep_enabled_)
        {
            left_motor_->disable();
            right_motor_->disable();
        }
        if (!l_ok || !r_ok)
        {
            RCLCPP_ERROR(logger(), "Zero-speed command not acknowledged (left=%s right=%s)",
                         l_ok ? "ok" : "FAILED", r_ok ? "ok" : "FAILED");
        }
    }
    catch (const std::exception & e)
    {
        RCLCPP_ERROR(logger(), "Exception while stopping motors: %s", e.what());
    }
    sent_left_rpm_  = 0;
    sent_right_rpm_ = 0;
}

// ============================================================
// export_state_interfaces
// ============================================================
std::vector<hardware_interface::StateInterface>
DraxHardware::export_state_interfaces()
{
    std::vector<hardware_interface::StateInterface> si;
    si.emplace_back(info_.joints[0].name, hardware_interface::HW_IF_POSITION, &left_wheel_position_);
    si.emplace_back(info_.joints[0].name, hardware_interface::HW_IF_VELOCITY, &left_wheel_velocity_);
    si.emplace_back(info_.joints[1].name, hardware_interface::HW_IF_POSITION, &right_wheel_position_);
    si.emplace_back(info_.joints[1].name, hardware_interface::HW_IF_VELOCITY, &right_wheel_velocity_);
    return si;
}

// ============================================================
// export_command_interfaces
// ============================================================
std::vector<hardware_interface::CommandInterface>
DraxHardware::export_command_interfaces()
{
    std::vector<hardware_interface::CommandInterface> ci;
    ci.emplace_back(info_.joints[0].name, hardware_interface::HW_IF_VELOCITY, &left_wheel_command_);
    ci.emplace_back(info_.joints[1].name, hardware_interface::HW_IF_VELOCITY, &right_wheel_command_);
    return ci;
}

// ============================================================
// read() — Python update_odometry()
//
// Reads encoders, applies all Python odometry corrections, updates
// theta (with IMU override), integrates position.
// ============================================================
hardware_interface::return_type
DraxHardware::read(const rclcpp::Time &, const rclcpp::Duration & period)
{
    // The torque service holds the CAN lock while it re-initialises the drives;
    // skip this cycle instead of blocking the controller_manager loop.
    std::unique_lock<std::mutex> lock(can_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
        return hardware_interface::return_type::OK;
    }
    try
    {
        return readImpl(period);
    }
    catch (const std::exception & e)
    {
        // e.g. diagnostics publish after rclcpp shutdown: never let it escape the RT loop
        RCLCPP_ERROR(logger(), "Exception in read(): %s", e.what());
        return hardware_interface::return_type::ERROR;
    }
}

hardware_interface::return_type
DraxHardware::readImpl(const rclcpp::Duration & period)
{
    // ---- Read raw encoder ticks ----
    // On a failed read keep the last raw value (last_*_ticks_ are post-swap).
    int32_t left_ticks_raw  = left_encoder_;
    int32_t right_ticks_raw = right_encoder_;

    if (!left_motor_->readEncoder(left_ticks_raw))
    {
        ++left_consecutive_read_failures_;
        RCLCPP_WARN(logger(), "Left encoder read failed (%d/%d)",
                    left_consecutive_read_failures_, MAX_CONSECUTIVE_READ_FAILURES);
        if (left_consecutive_read_failures_ > MAX_CONSECUTIVE_READ_FAILURES)
        {
            RCLCPP_ERROR(logger(), "Left encoder exceeded consecutive failure limit");
            return hardware_interface::return_type::ERROR;
        }
    }
    else
    {
        left_consecutive_read_failures_ = 0;
    }

    if (!right_motor_->readEncoder(right_ticks_raw))
    {
        ++right_consecutive_read_failures_;
        RCLCPP_WARN(logger(), "Right encoder read failed (%d/%d)",
                    right_consecutive_read_failures_, MAX_CONSECUTIVE_READ_FAILURES);
        if (right_consecutive_read_failures_ > MAX_CONSECUTIVE_READ_FAILURES)
        {
            RCLCPP_ERROR(logger(), "Right encoder exceeded consecutive failure limit");
            return hardware_interface::return_type::ERROR;
        }
    }
    else
    {
        right_consecutive_read_failures_ = 0;
    }

    // Store for balance controller encoder reads
    left_encoder_  = left_ticks_raw;
    right_encoder_ = right_ticks_raw;

    // ---- Python: swap_odometry_wheels ----
    int32_t left_ticks, right_ticks;
    if (swap_odometry_wheels_)
    {
        left_ticks  = right_ticks_raw;
        right_ticks = left_ticks_raw;
    }
    else
    {
        left_ticks  = left_ticks_raw;
        right_ticks = right_ticks_raw;
    }

    // ---- Timing ----
    // On first call, dt=0 — just initialise reference ticks
    const double dt = first_odom_ ? 0.0 : period.seconds();
    if (first_odom_)
    {
        // Reference ticks from initializeMotors() are raw (unswapped); re-base on the
        // swapped values so the first cycle yields zero motion instead of a jump.
        last_left_ticks_  = left_ticks;
        last_right_ticks_ = right_ticks;
    }
    first_odom_ = false;

    // ---- Tick deltas ----
    // Unsigned subtraction wraps correctly across the int32 encoder rollover.
    const int32_t delta_left_ticks  = static_cast<int32_t>(
        static_cast<uint32_t>(left_ticks)  - static_cast<uint32_t>(last_left_ticks_));
    const int32_t delta_right_ticks = static_cast<int32_t>(
        static_cast<uint32_t>(right_ticks) - static_cast<uint32_t>(last_right_ticks_));

    // ---- Python: ticks_to_distance ----
    double left_dist  = ticksToDistance(delta_left_ticks);
    double right_dist = ticksToDistance(delta_right_ticks);

    // ---- Python: invert_odometry_left / invert_odometry_right ----
    if (invert_odometry_left_)  { left_dist  = -left_dist;  }
    if (invert_odometry_right_) { right_dist = -right_dist; }

    // ---- Python: invert_odometry_linear ----
    if (invert_odometry_linear_)
    {
        left_dist  = -left_dist;
        right_dist = -right_dist;
    }

    // ---- Python: center distance ----
    double center_dist;
    if (swap_angular_linear_odometry_)
    {
        // Legacy compatibility only
        center_dist = right_dist - left_dist;
    }
    else
    {
        center_dist = (left_dist + right_dist) / 2.0;
    }

    // ---- Python: encoder_delta_theta ----
    double encoder_delta_theta = (right_dist - left_dist) / wheel_distance_;
    if (invert_odometry_angular_)
    {
        encoder_delta_theta = -encoder_delta_theta;
    }

    // ---- Python: IMU yaw override ----
    const double previous_theta = theta_;
    const auto   imu_yaw        = getFreshImuYaw();

    double new_theta;
    double delta_theta;

    if (imu_yaw.has_value())
    {
        new_theta   = imu_yaw.value();
        delta_theta = shortestAngleDelta(new_theta, previous_theta);
    }
    else
    {
        delta_theta = encoder_delta_theta;
        new_theta   = normalizeAngle(previous_theta + delta_theta);
    }

    // ---- Python: mid-heading integration (reduces arc error) ----
    const double heading_mid = normalizeAngle(previous_theta + 0.5 * delta_theta);
    const double delta_x     = center_dist * std::cos(heading_mid);
    const double delta_y     = center_dist * std::sin(heading_mid);

    x_     += delta_x;
    y_     += delta_y;
    theta_  = new_theta;

    // ---- Store for next iteration ----
    last_left_ticks_  = left_ticks;
    last_right_ticks_ = right_ticks;

    // ---- Measured wheel RPM (Python: for balance controller feedback) ----
    if (dt > 0.0)
    {
        double l_rpm = (static_cast<double>(delta_left_ticks)  / ticks_per_rev_) * (60.0 / dt);
        double r_rpm = (static_cast<double>(delta_right_ticks) / ticks_per_rev_) * (60.0 / dt);

        // Apply same inversions as distance
        if (invert_odometry_left_)    { l_rpm = -l_rpm; }
        if (invert_odometry_right_)   { r_rpm = -r_rpm; }
        if (invert_odometry_linear_)  { l_rpm = -l_rpm; r_rpm = -r_rpm; }

        current_left_rpm_  = l_rpm;
        current_right_rpm_ = r_rpm;
    }

    // ---- Update ros2_control state interfaces ----
    // Position in radians (full revolution = 2π)
    const double left_delta_rad  = (static_cast<double>(delta_left_ticks)  / ticks_per_rev_)
                                   * 2.0 * M_PI;
    const double right_delta_rad = (static_cast<double>(delta_right_ticks) / ticks_per_rev_)
                                   * 2.0 * M_PI;

    // Apply motor inversions to match what was commanded
    const double left_sign  = invert_left_  ? -1.0 : 1.0;
    const double right_sign = invert_right_ ? -1.0 : 1.0;
    left_wheel_position_  += left_sign  * left_delta_rad;
    right_wheel_position_ += right_sign * right_delta_rad;

    if (dt > 0.0)
    {
        left_wheel_velocity_  = left_sign  * left_delta_rad  / dt;
        right_wheel_velocity_ = right_sign * right_delta_rad / dt;
    }
    else
    {
        left_wheel_velocity_  = 0.0;
        right_wheel_velocity_ = 0.0;
    }

    // ---- Diagnostics ----
    if (diagnostics_updater_ && diagnostics_node_ &&
        (diagnostics_node_->now() - last_diagnostics_update_).seconds() >= diagnostics_period_sec_)
    {
        diagnostics_updater_->force_update();
        last_diagnostics_update_ = diagnostics_node_->now();
    }

    RCLCPP_DEBUG(logger(),
                 "read: ticks L=%d R=%d | dist L=%.4f R=%.4f | "
                 "x=%.3f y=%.3f theta=%.3f | RPM L=%.1f R=%.1f",
                 left_ticks, right_ticks,
                 left_dist, right_dist,
                 x_, y_, theta_,
                 current_left_rpm_, current_right_rpm_);

    return hardware_interface::return_type::OK;
}

// ============================================================
// write() — Python control_loop() motor write section.
//
// Pipeline (matches Python exactly):
//   command interfaces (rad/s) → RPM → clamp
//   → emergency stop / timeout check
//   → balance control
//   → slew rate limiter
//   → motor inversion
//   → CAN write
// ============================================================
hardware_interface::return_type
DraxHardware::write(const rclcpp::Time &, const rclcpp::Duration &)
{
    std::unique_lock<std::mutex> lock(can_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
        return hardware_interface::return_type::OK;
    }
    try
    {
        return writeImpl();
    }
    catch (const std::exception & e)
    {
        RCLCPP_ERROR(logger(), "Exception in write(): %s", e.what());
        return hardware_interface::return_type::ERROR;
    }
}

hardware_interface::return_type
DraxHardware::writeImpl()
{
    // ---- Torque disable guard ----
    if (torque_disabled_.load())
    {
        // Don't send any motor commands; encoders still read in read()
        return hardware_interface::return_type::OK;
    }

    // ---- Convert ros2_control wheel velocity (rad/s) → linear velocity (m/s) ----
    // rad/s * wheel_radius = m/s, where wheel_radius = wheel_circumference / (2π)
    // NaN/inf from a controller would become an undefined int cast further down.
    const double left_cmd_rad  = std::isfinite(left_wheel_command_)  ? left_wheel_command_  : 0.0;
    const double right_cmd_rad = std::isfinite(right_wheel_command_) ? right_wheel_command_ : 0.0;
    double left_vel  = left_cmd_rad  * (wheel_circumference_ / (2.0 * M_PI));
    double right_vel = right_cmd_rad * (wheel_circumference_ / (2.0 * M_PI));

    // ---- Python cmd_vel_callback() pipeline ----
    // The diff_drive_controller already computed left/right wheel m/s from
    // cmd_vel (linear.x, angular.z) using standard diff drive kinematics.
    // But Python motor.py applies additional inversions/swaps BEFORE kinematics.
    // We reproduce that by decomposing back, applying corrections, recomposing.

    // Decompose wheel velocities back to linear + angular components
    // Standard diff drive: left = linear - angular*D/2, right = linear + angular*D/2
    // So: linear = (left + right) / 2, angular = (right - left) / D
    double linear_vel  = (left_vel + right_vel) / 2.0;
    double angular_vel = (right_vel - left_vel) / wheel_distance_;

    // Python: swap_angular_linear_cmd_vel
    if (swap_angular_linear_cmd_vel_)
    {
        const double orig_linear  = linear_vel;
        const double orig_angular = angular_vel;
        linear_vel  = orig_angular * (wheel_distance_ / 2.0);
        angular_vel = orig_linear  * 2.0 / wheel_distance_;
    }

    // Python: invert_cmd_vel_linear
    if (invert_cmd_vel_linear_)
    {
        linear_vel = -linear_vel;
    }

    // Python: invert_cmd_vel_angular  (DEFAULT TRUE in Python!)
    if (invert_cmd_vel_angular_)
    {
        angular_vel = -angular_vel;
    }

    // Python: clamp to max velocities
    linear_vel  = clamp(linear_vel,  -max_linear_vel_,  max_linear_vel_);
    angular_vel = clamp(angular_vel, -max_angular_vel_, max_angular_vel_);

    // Recompose to wheel velocities using diff drive kinematics
    // Python: left_vel = linear_vel - (angular_vel * wheel_distance / 2.0)
    //         right_vel = linear_vel + (angular_vel * wheel_distance / 2.0)
    left_vel  = linear_vel - (angular_vel * wheel_distance_ / 2.0);
    right_vel = linear_vel + (angular_vel * wheel_distance_ / 2.0);

    // Python: swap_cmd_vel_wheels
    if (swap_cmd_vel_wheels_)
    {
        std::swap(left_vel, right_vel);
    }

    // Convert m/s → RPM: rpm = (vel * 60) / wheel_circumference
    const double left_rpm_f  = linearVelocityToRpm(left_vel);
    const double right_rpm_f = linearVelocityToRpm(right_vel);

    // Clamp to max RPM
    target_left_rpm_  = clamp(static_cast<int>(left_rpm_f),  -max_rpm_, max_rpm_);
    target_right_rpm_ = clamp(static_cast<int>(right_rpm_f), -max_rpm_, max_rpm_);

    // Update command timestamp if non-zero command
    if (target_left_rpm_ != 0 || target_right_rpm_ != 0)
    {
        last_motor_command_time_ = std::chrono::steady_clock::now();
    }

    // ---- Python: emergency stop ----
    // Flag is set by the subscriber thread; the RPM state is only touched here.
    if (emergency_stop_active_.load())
    {
        target_left_rpm_  = 0;
        target_right_rpm_ = 0;
        sent_left_rpm_    = 0;   // bypass the decel ramp: stop now
        sent_right_rpm_   = 0;
    }

    // ---- Python: motor command timeout ----
    const auto now_steady = std::chrono::steady_clock::now();
    const double time_since_cmd = std::chrono::duration<double>(
        now_steady - last_motor_command_time_).count();

    if (time_since_cmd > motor_command_timeout_)
    {
        if (target_left_rpm_ != 0 || target_right_rpm_ != 0)
        {
            RCLCPP_DEBUG(logger(), "Motor command timeout (%.2fs > %.2fs), stopping",
                         time_since_cmd, motor_command_timeout_);
            target_left_rpm_  = 0;
            target_right_rpm_ = 0;
        }
    }

    // ---- Python: apply_balance_control() ----
    auto [cmd_left, cmd_right, correction] =
        applyBalanceControl(target_left_rpm_, target_right_rpm_);

    // ---- Python: slew rate limiter ----
    const double dt_ramp = std::chrono::duration<double>(
        now_steady - last_ramp_time_).count();
    last_ramp_time_ = now_steady;

    if (ramp_enabled_)
    {
        cmd_left  = slewLimit(sent_left_rpm_,  cmd_left,
                              accel_rpm_per_sec_, decel_rpm_per_sec_, dt_ramp);
        cmd_right = slewLimit(sent_right_rpm_, cmd_right,
                              accel_rpm_per_sec_, decel_rpm_per_sec_, dt_ramp);
    }

    sent_left_rpm_  = cmd_left;
    sent_right_rpm_ = cmd_right;

    // ---- Python: TwoMotorController.set_pair() ----
    // (applies motor inversion then writes)
    const bool ok = motorSetPair(sent_left_rpm_, sent_right_rpm_, precise_speed_command_);

    // ---- Python: update balance reference when stopped ----
    if (balance_enabled_ && target_left_rpm_ == 0 && target_right_rpm_ == 0)
    {
        balance_ref_left_  = left_encoder_;
        balance_ref_right_ = right_encoder_;
    }

    RCLCPP_DEBUG(logger(),
                 "write: target L=%d R=%d | cmd L=%d R=%d | sent L=%d R=%d | "
                 "corr=%d | e-stop=%s | timeout=%.2fs",
                 target_left_rpm_, target_right_rpm_,
                 cmd_left, cmd_right,
                 sent_left_rpm_, sent_right_rpm_,
                 correction,
                 emergency_stop_active_.load() ? "YES" : "no",
                 time_since_cmd);

    if (!ok)
    {
        write_consecutive_failures_++;
        RCLCPP_WARN(logger(), "CAN motor write failed (%d/%d)",
                    write_consecutive_failures_, MAX_CONSECUTIVE_WRITE_FAILURES);
        if (write_consecutive_failures_ >= MAX_CONSECUTIVE_WRITE_FAILURES)
        {
            RCLCPP_ERROR(logger(), "CAN motor write failed %d consecutive times",
                         MAX_CONSECUTIVE_WRITE_FAILURES);
            return hardware_interface::return_type::ERROR;
        }
    }
    else
    {
        write_consecutive_failures_ = 0;
    }

    return hardware_interface::return_type::OK;
}

// ============================================================
// Python _slew_limit() — exact port
// ============================================================
int DraxHardware::slewLimit(int current, int target,
                            double accel_rate, double decel_rate, double dt)
{
    // Python:
    //   if dt <= 0.0 or current == target: return int(target)
    if (dt <= 0.0 || current == target)
    {
        return target;
    }

    const int    delta              = target - current;
    const bool   increasing_mag    = std::abs(target) > std::abs(current);
    const double rate              = increasing_mag ? accel_rate : decel_rate;
    const double max_step          = std::max(0.0, rate) * dt;

    // Python:
    //   if max_step <= 0.0: return int(current)
    if (max_step <= 0.0)
    {
        return current;
    }

    // Python:
    //   if abs(delta) <= max_step: return int(target)
    if (std::abs(delta) <= max_step)
    {
        return target;
    }

    // Python:
    //   step = int(max_step) if delta > 0 else -int(max_step)
    //   if step == 0:
    //     step = 1 if delta > 0 else -1
    int step = (delta > 0) ? static_cast<int>(max_step) : -static_cast<int>(max_step);
    if (step == 0)
    {
        step = (delta > 0) ? 1 : -1;
    }

    return current + step;
}

// ============================================================
// Python normalize_angle() — exact port
// ============================================================
double DraxHardware::normalizeAngle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

// ============================================================
// Python shortest_angle_delta() — exact port
// ============================================================
double DraxHardware::shortestAngleDelta(double new_angle, double old_angle)
{
    return std::atan2(std::sin(new_angle - old_angle),
                      std::cos(new_angle - old_angle));
}

// ============================================================
// Python ticks_to_distance() — exact port
// ============================================================
double DraxHardware::ticksToDistance(int32_t ticks) const
{
    return (static_cast<double>(ticks) / ticks_per_rev_) * wheel_circumference_;
}

// ============================================================
// Python rpm_to_linear_velocity()
// ============================================================
double DraxHardware::rpmToLinearVelocity(double rpm) const
{
    return (rpm * wheel_circumference_) / 60.0;
}

// ============================================================
// Python linear_velocity_to_rpm()
// ============================================================
double DraxHardware::linearVelocityToRpm(double linear_vel) const
{
    return (linear_vel * 60.0) / wheel_circumference_;
}

// ============================================================
// Python get_fresh_imu_yaw() — exact port
// ============================================================
std::optional<double> DraxHardware::getFreshImuYaw()
{
    if (!use_imu_yaw_)
    {
        return std::nullopt;
    }

    double yaw, gyro_z, stamp_sec;
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (!imu_available_)
        {
            return std::nullopt;
        }
        yaw       = imu_theta_;
        gyro_z    = imu_angular_velocity_z_;
        stamp_sec = imu_stamp_sec_;
    }

    // Python: age = get_ros_time_sec() - stamp_sec
    const double now_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    // Use wall clock relative age: since imu_stamp_sec_ is set from steady clock too
    double age = now_sec - stamp_sec;
    if (age < 0.0) { age = 0.0; }

    if (age > imu_yaw_timeout_)
    {
        RCLCPP_WARN_THROTTLE(logger(), *rclcpp::Clock::make_shared(), 2000,
                             "IMU yaw is stale (%.3fs old); falling back to encoder yaw.", age);
        return std::nullopt;
    }

    // Python: if use_imu_gyro_extrapolation:
    //   yaw += gyro_z * min(age, max(0.0, max_imu_extrapolation))
    if (use_imu_gyro_extrapolation_)
    {
        yaw += gyro_z * std::min(age, std::max(0.0, max_imu_extrapolation_));
    }

    return normalizeAngle(yaw);
}

// ============================================================
// Python get_current_yaw() — exact port
// ============================================================
double DraxHardware::getCurrentYaw()
{
    const auto imu_yaw = getFreshImuYaw();
    if (imu_yaw.has_value())
    {
        RCLCPP_INFO_THROTTLE(logger(), *rclcpp::Clock::make_shared(), 1000,
                             "YAW SOURCE: IMU | Yaw: %.4f rad", imu_yaw.value());
        return imu_yaw.value();
    }
    else
    {
        RCLCPP_INFO_THROTTLE(logger(), *rclcpp::Clock::make_shared(), 1000,
                             "YAW SOURCE: ENCODER | Yaw: %.4f rad", theta_);
        return theta_;
    }
}

// ============================================================
// Python apply_balance_control() — enhanced version (exact port)
// ============================================================
std::tuple<int, int, int>
DraxHardware::applyBalanceControl(int target_left, int target_right)
{
    if (!balance_enabled_)
    {
        return {target_left, target_right, 0};
    }

    try
    {
        // ---- 1. READ ENCODERS (position) ----
        // Use the already-read encoder values from read() to avoid extra CAN traffic
        const int32_t enc_left  = left_encoder_;
        const int32_t enc_right = right_encoder_;

        const int32_t progress_left  = std::abs(enc_left  - balance_ref_left_);
        const int32_t progress_right = std::abs(enc_right - balance_ref_right_);
        const int32_t encoder_error  = progress_left - progress_right;

        // ---- 2. HEADING CONTROL (IMU) ----
        const double current_yaw = getCurrentYaw();

        // Python: if not hasattr(self, "target_yaw"): self.target_yaw = current_yaw
        if (!target_yaw_initialized_)
        {
            target_yaw_             = current_yaw;
            target_yaw_initialized_ = true;
        }

        // Python: yaw_error = atan2(sin(target_yaw - current_yaw), cos(target_yaw - current_yaw))
        double yaw_error = target_yaw_ - current_yaw;
        yaw_error = std::atan2(std::sin(yaw_error), std::cos(yaw_error));

        const double kp_heading         = 0.5;
        const double heading_correction = kp_heading * yaw_error;

        // ---- 3. SPEED CONTROL (RPM feedback) ----
        const double left_error  = static_cast<double>(target_left)  - current_left_rpm_;
        const double right_error = static_cast<double>(target_right) - current_right_rpm_;

        const double kp_speed        = 0.2;
        const double speed_corr_left  = kp_speed * left_error;
        const double speed_corr_right = kp_speed * right_error;

        // ---- 4. ORIGINAL ENCODER BALANCE ----
        const double avg_speed    = (std::abs(target_left) + std::abs(target_right)) / 2.0;
        const int    dynamic_limit = static_cast<int>(
            std::max(5.0, avg_speed * clamp(safe_balance_ratio_, 0.0, 1.0)));
        const int corr_limit = std::min(max_correction_rpm_, dynamic_limit);

        const int encoder_correction = clamp(
            static_cast<int>(balance_kp_ * static_cast<double>(encoder_error)),
            -corr_limit, corr_limit);

        // ---- 5. COMBINE ALL CONTROLS ----
        // Python:
        //   left_cmd  = target_left  + speed_corr_left  - heading_correction
        //   right_cmd = target_right + speed_corr_right + heading_correction
        double left_cmd  = static_cast<double>(target_left)  + speed_corr_left  - heading_correction;
        double right_cmd = static_cast<double>(target_right) + speed_corr_right + heading_correction;

        // Apply encoder balancing on top
        if (static_cast<long long>(target_left) * target_right >= 0)  // straight motion
        {
            if (encoder_error > 0)
            {
                right_cmd += encoder_correction;
            }
            else
            {
                left_cmd += encoder_correction;
            }
        }
        else  // rotation case
        {
            left_cmd  -= encoder_correction;
            right_cmd += encoder_correction;
        }

        // ---- 6. CLAMP FINAL OUTPUT ----
        const int final_left  = clamp(static_cast<int>(left_cmd),  -max_rpm_, max_rpm_);
        const int final_right = clamp(static_cast<int>(right_cmd), -max_rpm_, max_rpm_);

        RCLCPP_DEBUG(logger(),
                     "balance: enc_err=%d  yaw_err=%.4f  heading_corr=%.2f  "
                     "enc_corr=%d  L=%d  R=%d",
                     encoder_error, yaw_error, heading_correction,
                     encoder_correction, final_left, final_right);

        return {final_left, final_right, static_cast<int>(heading_correction)};
    }
    catch (const std::exception & e)
    {
        RCLCPP_ERROR(logger(), "Error in balance control: %s", e.what());
        return {target_left, target_right, 0};
    }
}

// ============================================================
// readEncoders — read both motors in order (left then right)
// ============================================================
bool DraxHardware::readEncoders(int32_t & left, int32_t & right)
{
    return left_motor_->readEncoder(left) && right_motor_->readEncoder(right);
}

// ============================================================
// Python TwoMotorController.set_pair()
//   Applies inversion then writes.
// ============================================================
bool DraxHardware::motorSetPair(int left_rpm, int right_rpm, bool precise)
{
    const int out_left  = invert_left_  ? -left_rpm  : left_rpm;
    const int out_right = invert_right_ ? -right_rpm : right_rpm;

    RCLCPP_DEBUG(logger(),
                 "motorSetPair: cmd L=%d R=%d → can L=%d R=%d (precise=%s)",
                 left_rpm, right_rpm, out_left, out_right,
                 precise ? "yes" : "no");

    const bool l_ok = left_motor_->setRPM(out_left,  precise);
    const bool r_ok = right_motor_->setRPM(out_right, precise);
    return l_ok && r_ok;
}

// ============================================================
// Python TwoMotorController.stop() = set_pair(0,0,precise=true)
// ============================================================
bool DraxHardware::motorStop()
{
    return motorSetPair(0, 0, true);
}

// ============================================================
// Python TwoMotorController.disable()
// ============================================================
void DraxHardware::motorDisable()
{
    if (left_motor_)  { left_motor_->disable();  }
    if (right_motor_) { right_motor_->disable(); }
}

// ============================================================
// Python DifferentialDriveController.initialize_motors()
// ============================================================
bool DraxHardware::initializeMotors()
{
    try
    {
        // Python:
        //   if nmt_start: io.send_nmt_start(0); time.sleep(0.15)
        if (nmt_start_)
        {
            RCLCPP_INFO(logger(), "Sending NMT Start (node 0)");
            if (!bus_->sendNMTStart(0))
            {
                RCLCPP_WARN(logger(), "Failed to send NMT start");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }

        // Python: controller.init(mode, fix_din3)
        //   → left.init_speed_mode(mode, fix_din3)
        //   → right.init_speed_mode(mode, fix_din3)
        RCLCPP_INFO(logger(), "Initializing left motor (mode=%d, fix_din3=%s)",
                    motor_mode_, fix_din3_ ? "yes" : "no");
        if (!left_motor_->initSpeedMode(motor_mode_, fix_din3_))
        {
            RCLCPP_ERROR(logger(), "Failed to initialize left motor");
            return false;
        }

        RCLCPP_INFO(logger(), "Initializing right motor (mode=%d, fix_din3=%s)",
                    motor_mode_, fix_din3_ ? "yes" : "no");
        if (!right_motor_->initSpeedMode(motor_mode_, fix_din3_))
        {
            RCLCPP_ERROR(logger(), "Failed to initialize right motor");
            return false;
        }

        // Python: controller.stop()
        RCLCPP_INFO(logger(), "Stopping motors after init");
        if (!motorStop())
        {
            RCLCPP_ERROR(logger(), "Failed to stop motors after init");
            return false;
        }

        // Python: balance_ref_left, balance_ref_right = controller.read_encoders()
        //         last_left_ticks = balance_ref_left
        //         last_right_ticks = balance_ref_right
        int32_t enc_left  = 0;
        int32_t enc_right = 0;
        if (!readEncoders(enc_left, enc_right))
        {
            RCLCPP_ERROR(logger(), "Failed to read encoders after motor init");
            return false;
        }

        balance_ref_left_  = enc_left;
        balance_ref_right_ = enc_right;
        last_left_ticks_   = enc_left;
        last_right_ticks_  = enc_right;
        left_encoder_      = enc_left;
        right_encoder_     = enc_right;

        // Python:
        //   sent_left_rpm = 0; sent_right_rpm = 0
        //   last_ramp_time = time.time()
        sent_left_rpm_   = 0;
        sent_right_rpm_  = 0;
        last_ramp_time_  = std::chrono::steady_clock::now();

        RCLCPP_INFO(logger(), "Motors initialized. Encoder L=%d R=%d",
                    enc_left, enc_right);
        return true;
    }
    catch (const std::exception & e)
    {
        RCLCPP_ERROR(logger(), "Motor initialization failed: %s", e.what());
        return false;
    }
}

// ============================================================
// Python DifferentialDriveController.shutdown_motors()
// ============================================================
void DraxHardware::shutdownMotors()
{
    try
    {
        // Python: controller.stop()
        if (left_motor_ && right_motor_)
        {
            motorStop();
        }

        // Python: if not keep_enabled: controller.disable()
        if (!keep_enabled_ && left_motor_ && right_motor_)
        {
            motorDisable();
        }
        RCLCPP_INFO(logger(), "Motors shut down.");
    }
    catch (const std::exception & e)
    {
        RCLCPP_ERROR(logger(), "Error during motor shutdown: %s", e.what());
    }
}

// ============================================================
// Python _apply_motor_runtime_config()
// ============================================================
bool DraxHardware::applyMotorRuntimeConfig()
{
    // Python:
    //   target_left_rpm = target_right_rpm = 0
    //   sent_left_rpm = sent_right_rpm = 0
    //   last_motor_command_time = time.time()
    //   last_ramp_time = time.time()
    //
    //   controller.stop()
    //   controller.left.node.timeout = timeout_sec
    //   controller.right.node.timeout = timeout_sec
    //
    //   if nmt_start: io.send_nmt_start(0); sleep(0.15)
    //
    //   controller.init(mode, fix_din3)
    //   controller.stop()
    //
    //   balance_ref_left, balance_ref_right = read_encoders()
    //   last_left_ticks = balance_ref_left
    //   last_right_ticks = balance_ref_right

    target_left_rpm_      = 0;
    target_right_rpm_     = 0;
    sent_left_rpm_        = 0;
    sent_right_rpm_       = 0;
    last_motor_command_time_ = std::chrono::steady_clock::now();
    last_ramp_time_          = std::chrono::steady_clock::now();

    // Update SDO timeout on existing nodes
    sdo_timeout_ = std::chrono::milliseconds(
        static_cast<int>(timeout_sec_ * 1000.0));

    if (!motorStop()) { return false; }

    if (nmt_start_)
    {
        if (!bus_->sendNMTStart(0))
        {
            RCLCPP_WARN(logger(), "NMT start failed during runtime config apply");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }

    if (!left_motor_->initSpeedMode(motor_mode_, fix_din3_))  { return false; }
    if (!right_motor_->initSpeedMode(motor_mode_, fix_din3_)) { return false; }
    if (!motorStop()) { return false; }

    int32_t enc_left = 0, enc_right = 0;
    if (!readEncoders(enc_left, enc_right)) { return false; }

    balance_ref_left_  = enc_left;
    balance_ref_right_ = enc_right;
    last_left_ticks_   = enc_left;
    last_right_ticks_  = enc_right;

    RCLCPP_INFO(logger(), "Applied motor runtime config: mode=%d timeout=%.3fs",
                motor_mode_, timeout_sec_);
    return true;
}

// ============================================================
// IMU callback — Python imu_callback()
// ============================================================
void DraxHardware::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    try
    {
        // Python:
        //   q = msg.orientation
        //   siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        //   cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        //   yaw = atan2(siny_cosp, cosy_cosp)
        //   yaw = atan2(sin(yaw), cos(yaw))   # normalize
        //   if invert_imu_yaw: yaw = -yaw

        const auto & q = msg->orientation;
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        // normalize
        yaw = std::atan2(std::sin(yaw), std::cos(yaw));

        double gyro_z = msg->angular_velocity.z;

        if (invert_imu_yaw_)
        {
            yaw    = -yaw;
            gyro_z = -gyro_z;
        }

        // Python: stamp_sec = stamp_to_sec(msg.header.stamp)
        //         imu_stamp_sec = stamp_sec if stamp_sec > 0.0 else get_ros_time_sec()
        const double stamp_sec =
            static_cast<double>(msg->header.stamp.sec) +
            static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

        // For age calculation we use steady_clock in this implementation
        // Use wall time from the stamp if valid, else use current steady clock
        const double now_steady = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        const double used_stamp = (stamp_sec > 0.0)
            ? (now_steady - 0.0)  // age will be ~0 for fresh messages
            : now_steady;
        (void)used_stamp;

        // Store the steady-clock time of this IMU message for age calculation
        const double imu_recv_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            imu_theta_              = yaw;
            imu_angular_velocity_z_ = gyro_z;
            imu_stamp_sec_          = imu_recv_time;
            imu_available_          = true;
        }
    }
    catch (const std::exception & e)
    {
        RCLCPP_ERROR(logger(), "Error processing IMU data: %s", e.what());
    }
}

// ============================================================
// ============================================================
// Torque disable service callback
// ============================================================
void DraxHardware::handleSetTorqueDisable(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
    std::shared_ptr<std_srvs::srv::SetBool::Response> res)
{
    // Blocks the control loop's CAN access (read/write skip cycles) while this runs.
    std::lock_guard<std::mutex> lock(can_mutex_);
    if (!bus_ || !bus_->isOpen() || !left_motor_ || !right_motor_)
    {
        res->success = false;
        res->message = "CAN bus not open";
        return;
    }

    torque_disabled_.store(req->data);
    if (req->data)
    {
        // Immediately stop and disable motors
        motorStop();
        left_motor_->disable();
        right_motor_->disable();
        RCLCPP_INFO(logger(),
                     "Torque DISABLED — motors free-wheeling, encoders still reading.");
    }
    else
    {
        // Re-initialize motors to restore speed mode
        setSdoPolicy(false);
        const bool ok = initializeMotors();
        setSdoPolicy(true);
        // Odometry: initializeMotors() re-bases last_*_ticks_ on raw (unswapped) encoders
        first_odom_ = true;
        if (!ok)
        {
            res->success = false;
            res->message = "Failed to re-initialize motors";
            RCLCPP_ERROR(logger(), "Failed to re-enable torque: motor init failed");
            return;
        }
        RCLCPP_INFO(logger(), "Torque ENABLED — motors back in speed control.");
    }
    res->success = true;
    res->message = req->data ? "Torque disabled" : "Torque enabled";
}

// Emergency stop callback — Python emergency_stop_callback()
// ============================================================
void DraxHardware::emergencyStopCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    const bool active = msg->data;
    emergency_stop_active_.store(active);

    // Runs on the internal spin thread: only touch the atomic flag here,
    // write() zeroes the RPM state on the control thread.
    if (active)
    {
        RCLCPP_WARN(logger(), "Emergency stop ACTIVATED. Motor commands forced to zero.");
    }
    else
    {
        RCLCPP_INFO(logger(), "Emergency stop CLEARED. Motion commands accepted again.");
    }
}

// ============================================================
// Internal node (for subscriptions)
// ============================================================
void DraxHardware::startInternalNode()
{
    internal_node_ = rclcpp::Node::make_shared("drax_hw_internal");
    internal_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    internal_executor_->add_node(internal_node_);

    // IMU subscription
    if (use_imu_yaw_)
    {
        auto qos = rclcpp::QoS(10).best_effort();
        imu_sub_ = internal_node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                this->imuCallback(msg);
            });
        RCLCPP_INFO(logger(), "Subscribed to IMU topic: %s", imu_topic_.c_str());
    }

    // Emergency stop subscription (Python: reliable QoS)
    auto reliable_qos = rclcpp::QoS(10).reliable();
    emergency_stop_sub_ = internal_node_->create_subscription<std_msgs::msg::Bool>(
        emergency_stop_topic_, reliable_qos,
        [this](const std_msgs::msg::Bool::SharedPtr msg) {
            this->emergencyStopCallback(msg);
        });
    RCLCPP_INFO(logger(), "Subscribed to emergency stop topic: %s",
                emergency_stop_topic_.c_str());

    // Torque disable service
    torque_disable_srv_ = internal_node_->create_service<std_srvs::srv::SetBool>(
        "/drax/set_torque_disable",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
            this->handleSetTorqueDisable(req, res);
        });
    RCLCPP_INFO(logger(), "Torque disable service available: /drax/set_torque_disable");

    // Spin in background thread
    internal_spin_thread_ = std::thread([this]() {
        RCLCPP_DEBUG(logger(), "Internal spin thread started");
        internal_executor_->spin();
        RCLCPP_DEBUG(logger(), "Internal spin thread stopped");
    });
}

void DraxHardware::stopInternalNode()
{
    if (internal_executor_)
    {
        internal_executor_->cancel();
    }
    if (internal_spin_thread_.joinable())
    {
        internal_spin_thread_.join();
    }
    imu_sub_.reset();
    emergency_stop_sub_.reset();
    torque_disable_srv_.reset();
    internal_executor_.reset();
    internal_node_.reset();
}

// ============================================================
// Diagnostics
// ============================================================
void DraxHardware::updateDiagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Motors communicating");
    addMotorDiagnostics(stat, "left",  left_motor_.get());
    addMotorDiagnostics(stat, "right", right_motor_.get());
}

void DraxHardware::addMotorDiagnostics(
    diagnostic_updater::DiagnosticStatusWrapper & stat,
    const std::string & name,
    Motor * motor)
{
    if (!motor)
    {
        stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                          name + " motor unavailable");
        return;
    }

    uint16_t status = 0;
    if (motor->readStatusWord(status))
    {
        stat.add(name + "_status_word", status);
    }
    else
    {
        stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                          name + " motor status read failed");
    }

    double temperature = 0.0;
    if (motor->readTemperature(temperature))
    {
        stat.add(name + "_temperature", temperature);
    }
    else if (motor_config_.temperature_index != 0)
    {
        stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                          name + " motor temperature read failed");
    }

    double voltage = 0.0;
    if (motor->readBusVoltage(voltage))
    {
        stat.add(name + "_bus_voltage", voltage);
    }
    else if (motor_config_.voltage_index != 0)
    {
        stat.mergeSummary(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                          name + " motor bus voltage read failed");
    }
}

} // namespace drax_hardware

PLUGINLIB_EXPORT_CLASS(
    drax_hardware::DraxHardware,
    hardware_interface::SystemInterface)