#include <Arduino.h>
#include "HoldTransistorController.hpp"

HoldTransistorController bjt;

void setup()
{
}

void loop()
{
    // On
    bjt.set_transistor(HoldTransistorController::TRANSISTOR_ON);
    delay(10000);

    // Off
    bjt.set_transistor(HoldTransistorController::TRANSISTOR_OFF);
    delay(10000);
}
