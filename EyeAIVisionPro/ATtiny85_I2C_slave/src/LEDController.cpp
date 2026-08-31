#include "Arduino.h"
#include "configuration.h"

#include <LEDController.hpp>

LEDController::LEDController() {
    init();
}

void LEDController::init() {
    pinMode(LED_GREEN_ENABLE, OUTPUT);

    set_led(current_state);
}

void LEDController::set_led(LED_STATE new_state) {
    current_state = new_state;

    switch (current_state) {
        case GREEN_ON:       // only green on
            digitalWrite(LED_GREEN_ENABLE, HIGH);
            break;

        case RED_ON:         // only red on
            digitalWrite(LED_GREEN_ENABLE, LOW);
            break;
    }
}
