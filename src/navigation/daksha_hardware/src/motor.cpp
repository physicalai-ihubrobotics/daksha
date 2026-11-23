#include "drax_hardware/motor.hpp"

#include <thread>
#include <chrono>
#include <cmath>

#include <rclcpp/rclcpp.hpp>

namespace drax_hardware
{

Motor::Motor(CanOpenNode &node,
             const MotorConfig & config,
             const rclcpp::Logger & logger)
: node_(node),
  config_(config),
  logger_(logger)
{
}

// Python: int(round((rpm * 512 * feedback_resolution) / 1875))
int32_t Motor::rpmToDec(int rpm)
{
    return static_cast<int32_t>(
        std::round((rpm * 512.0 * config_.feedback_resolution) / 1875.0));
}

// Matches Python Motor.init_speed_mode(mode, fix_din3):
//   node.write(0x6060, 0x00, mode, size=1, signed=True)
//   node.write(0x6040, 0x00, 0x86, size=2, signed=True)
//   node.write(0x605A, 0x11, 0,    size=1, signed=False)
//   if fix_din3:
//     pol = read_u16(0x2010, 0x01); write(0x2010, 0x01, pol & ~(1<<2), 2, False)
//     sim = read_u16(0x2010, 0x02); write(0x2010, 0x02, sim & ~(1<<2), 2, False)
//   node.write(0x6040, 0x00, 0x06, size=2, signed=True)
//   time.sleep(0.02)
//   node.write(0x6040, 0x00, 0x07, size=2, signed=True)
//   time.sleep(0.02)
//   node.write(0x6040, 0x00, 0x0F, size=2, signed=True)
//   self.set_rpm(0, precise=True)
bool Motor::initSpeedMode(int8_t mode, bool fix_din3)
{
    RCLCPP_INFO(logger_, "Initializing speed mode (mode=%d, fix_din3=%s)",
                mode, fix_din3 ? "true" : "false");

    // Set operation mode
    RCLCPP_DEBUG(logger_, "Writing operation mode 0x6060:00 = %d", mode);
    if (!node_.write(0x6060, 0x00, static_cast<int32_t>(mode), 1, true))
    {
        RCLCPP_ERROR(logger_, "Failed to write operation mode 0x6060");
        return false;
    }

    // Control word: fault reset / quick-stop (0x86)
    RCLCPP_DEBUG(logger_, "Writing control word 0x6040:00 = 0x0086");
    if (!node_.write(0x6040, 0x00, 0x86, 2, true))
    {
        RCLCPP_ERROR(logger_, "Failed to write control word 0x0086");
        return false;
    }

    // Quick-stop option code (subindex 0x11 = 17)
    RCLCPP_DEBUG(logger_, "Writing quick-stop option code 0x605A:11 = 0");
    if (!node_.write(0x605A, 0x11, 0, 1, false))
    {
        RCLCPP_ERROR(logger_, "Failed to write quick-stop option code 0x605A:11");
        return false;
    }

    // DIN3 fix: clear bit 2 of polarity and simulation registers
    if (fix_din3)
    {
        RCLCPP_DEBUG(logger_, "Applying vendor DIN3 fix");

        uint16_t pol_val = 0;
        if (!node_.readU16(0x2010, 0x01, pol_val))
        {
            RCLCPP_ERROR(logger_, "Failed to read DIN polarity 0x2010:01");
            return false;
        }
        if (!node_.write(0x2010, 0x01,
                         static_cast<int32_t>(pol_val & ~static_cast<uint16_t>(1U << 2)),
                         2, false))
        {
            RCLCPP_ERROR(logger_, "Failed to write DIN polarity 0x2010:01");
            return false;
        }

        uint16_t sim_val = 0;
        if (!node_.readU16(0x2010, 0x02, sim_val))
        {
            RCLCPP_ERROR(logger_, "Failed to read DIN simulation 0x2010:02");
            return false;
        }
        if (!node_.write(0x2010, 0x02,
                         static_cast<int32_t>(sim_val & ~static_cast<uint16_t>(1U << 2)),
                         2, false))
        {
            RCLCPP_ERROR(logger_, "Failed to write DIN simulation 0x2010:02");
            return false;
        }
    }

    // State machine: Shutdown → Switch on → Enable operation
    RCLCPP_DEBUG(logger_, "Writing control word 0x6040:00 = 0x0006 (Shutdown)");
    if (!node_.write(0x6040, 0x00, 0x06, 2, true))
    {
        RCLCPP_ERROR(logger_, "Failed to write control word 0x0006");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    RCLCPP_DEBUG(logger_, "Writing control word 0x6040:00 = 0x0007 (Switch on)");
    if (!node_.write(0x6040, 0x00, 0x07, 2, true))
    {
        RCLCPP_ERROR(logger_, "Failed to write control word 0x0007");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    RCLCPP_DEBUG(logger_, "Writing control word 0x6040:00 = 0x000F (Enable operation)");
    if (!node_.write(0x6040, 0x00, 0x0F, 2, true))
    {
        RCLCPP_ERROR(logger_, "Failed to write control word 0x000F");
        return false;
    }

    RCLCPP_INFO(logger_, "Speed mode initialized");

    // Python: self.set_rpm(0, precise=True)
    return setRPM(0, true);
}

// Matches Python Motor.set_rpm(rpm, precise):
//   self.node.write(0x2FF0, 0x09, rpm, size=2, signed=True)
//   if precise:
//     self.node.write(0x60FF, 0x00, rpm_to_dec(rpm), size=4, signed=True)
bool Motor::setRPM(int rpm, bool precise)
{
    RCLCPP_DEBUG(logger_, "setRPM(%d, precise=%s)", rpm, precise ? "true" : "false");

    // Always write simple RPM command first
    if (!node_.write(0x2FF0, 0x09, static_cast<int32_t>(rpm), 2, true))
    {
        RCLCPP_ERROR(logger_, "Failed to write simple RPM 0x2FF0:09 = %d", rpm);
        return false;
    }

    if (precise)
    {
        const int32_t dec_val = rpmToDec(rpm);
        RCLCPP_DEBUG(logger_, "setRPM precise: 0x60FF:00 = %d (dec)", dec_val);
        if (!node_.write(0x60FF, 0x00, dec_val, 4, true))
        {
            RCLCPP_ERROR(logger_, "Failed to write precise velocity 0x60FF:00 = %d", dec_val);
            return false;
        }
    }

    return true;
}

// Python: Motor.stop() = self.set_rpm(0, precise=True)
bool Motor::stop()
{
    return setRPM(0, true);
}

// Python: Motor.disable() = self.node.write(0x6040, 0x00, 0x06, size=2, signed=True)
bool Motor::disable()
{
    return node_.write(0x6040, 0x00, 0x06, 2, true);
}

bool Motor::quickStop()
{
    RCLCPP_WARN(logger_, "Sending CANopen quick stop");
    return node_.write(0x6040, 0x00, 0x02, 2, true);
}

bool Motor::readEncoder(int32_t &encoder)
{
    return node_.readI32(0x6063, 0x00, encoder);
}

bool Motor::readStatusWord(uint16_t &status)
{
    return node_.readU16(0x6041, 0x00, status);
}

bool Motor::readTemperature(double &temperature)
{
    if (config_.temperature_index == 0)
    {
        return false;
    }
    int32_t raw = 0;
    if (!node_.readI32(config_.temperature_index, config_.temperature_subindex, raw))
    {
        return false;
    }
    temperature = raw * config_.temperature_scale;
    return true;
}

bool Motor::readBusVoltage(double &voltage)
{
    if (config_.voltage_index == 0)
    {
        return false;
    }
    uint32_t raw = 0;
    if (!node_.readU32(config_.voltage_index, config_.voltage_subindex, raw))
    {
        return false;
    }
    voltage = raw * config_.voltage_scale;
    return true;
}

} // namespace drax_hardware
