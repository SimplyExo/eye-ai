#pragma once

#include <Arduino.h>

/*  Class for reading the voltage at the resistor of the voltage divider in order to 
    estimate the battery level based on its voltage.
*/

class BatteryState {
    public:
        BatteryState();

        uint8_t read_analog();
        void init();
};
