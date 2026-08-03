#include <Arduino.h>
#include "LEDController.hpp"

LEDController led;

void setup()
{
}

void loop()
{
    // Green
    led.set_led(LEDController::GREEN_ON);
    delay(2000);

    // Red
    led.set_led(LEDController::RED_ON);
    delay(2000);
}
