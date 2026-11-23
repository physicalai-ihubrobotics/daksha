#pragma once

#include "drax_hardware/canopen_node.hpp"

#include <cstdint>

#include <rclcpp/logger.hpp>

namespace drax_hardware
{

struct MotorConfig
{
    int feedback_resolution = 4096;
    bool fix_din3 = true;
    uint16_t temperature_index = 0;
    uint8_t temperature_subindex = 0;
    double temperature_scale = 1.0;
    uint16_t voltage_index = 0;
    uint8_t voltage_subindex = 0;
    double voltage_scale = 1.0;
};

class Motor
{
public:

    Motor(CanOpenNode &node,
          const MotorConfig & config,
          const rclcpp::Logger & logger);

    // Matches Python Motor.init_speed_mode(mode, fix_din3)
    // SDO sequence: 0x6060, 0x6040=0x86, 0x605A:11=0,
    //               [DIN3 fix], 0x6040=0x06, sleep 20ms,
    //               0x6040=0x07, sleep 20ms, 0x6040=0x0F, set_rpm(0,true)
    bool initSpeedMode(int8_t mode = -3, bool fix_din3 = true);

    // Matches Python Motor.set_rpm(rpm, precise)
    // precise=false: writes only 0x2FF0:09 (int16)
    // precise=true:  writes 0x2FF0:09 (int16) then 0x60FF:00 (int32 dec)
    bool setRPM(int rpm, bool precise = true);

    // stop() = set_rpm(0, precise=true)
    bool stop();

    // disable() = write 0x6040:00 = 0x06
    bool disable();

    // quickStop() = write 0x6040:00 = 0x02
    bool quickStop();

    bool readEncoder(int32_t &encoder);

    bool readStatusWord(uint16_t &status);

    bool readTemperature(double &temperature);

    bool readBusVoltage(double &voltage);

private:

    // Python: int(round((rpm * 512 * feedback_resolution) / 1875))
    int32_t rpmToDec(int rpm);

    CanOpenNode &node_;

    MotorConfig config_;

    rclcpp::Logger logger_;
};

} // namespace drax_hardware
