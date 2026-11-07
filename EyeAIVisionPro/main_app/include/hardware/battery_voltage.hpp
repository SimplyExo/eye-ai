#pragma once

#include <cstdint>

#define BATTERY_DEV "/dev/battery"

#define R_12 100000.0 // Ohms
#define R_13 28000.0 // Ohms

class battery_voltage {
    public:
        battery_voltage() = default;

        double get_battery_voltage();
        double get_battery_percentage();
        double get_battery_percentage(double v);
    private:
        uint8_t get_raw();
        double digit_to_volt(uint8_t digit);

        double f0_80(double voltage);   // State of Charge Approximation for about 0% to 80%
        double f80_100(double voltage);   // State of Charge Approximation for about 80% to 100%
};

