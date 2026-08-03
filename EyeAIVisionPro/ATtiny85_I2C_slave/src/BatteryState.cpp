#include "Arduino.h"

#include "configuration.h"
#include <BatteryState.hpp>

BatteryState::BatteryState() {
    init();
}

void BatteryState::init() {
    pinMode(BATTERY_VOLTAGE, INPUT);
}

uint16_t BatteryState::read_analog() {
    return analogRead(BATTERY_VOLTAGE);
}
