#include "Arduino.h"

#include <configuration.h>
#include <HoldTransistorController.hpp>

HoldTransistorController::HoldTransistorController() {
    init();
};

void HoldTransistorController::init() {
    pinMode(HOLD_TRANSISTOR, OUTPUT);

    set_transistor(current_state);
}

void HoldTransistorController::set_transistor(TRANSISTOR_STATE new_state) {
    current_state = new_state;

    switch (current_state) {
        case TRANSISTOR_ON:
            digitalWrite(HOLD_TRANSISTOR, HIGH);    // switch on transistor
            break;

        case TRANSISTOR_OFF:
            digitalWrite(HOLD_TRANSISTOR, LOW);    // switch off transistor
            break;
    }
}
