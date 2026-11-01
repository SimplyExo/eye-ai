#include "Arduino.h"

#include "configuration.h"
#include <BatteryState.hpp>

BatteryState::BatteryState() = default;

void BatteryState::init() {
    pinMode(BATTERY_VOLTAGE, INPUT);
}

uint8_t BatteryState::read_analog() {
    return (uint8_t)(analogRead(BATTERY_VOLTAGE) >> 2);
}
