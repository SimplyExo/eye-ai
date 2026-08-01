#include "Arduino.h"
#include "configuration.h"

#include <LEDController.hpp>

LEDController::LEDController() {
    init();
}

void LEDController::init() {
    pinMode(LED_R, OUTPUT);
    pinMode(LED_G, OUTPUT);

    set_led(current_state);
}

void LEDController::set_led(LED_STATE new_state) {
    current_state = new_state;

    switch (current_state) {
        case GREEN_ON:       // only green on
            digitalWrite(LED_R, LOW);
            digitalWrite(LED_G, HIGH);
            break;

        case RED_ON:         // only red on
            digitalWrite(LED_G, LOW);
            digitalWrite(LED_R, HIGH);
            break;
            
        case OFF:            // completely off
            digitalWrite(LED_R, LOW);
            digitalWrite(LED_G, LOW);
            break;
    }
}
