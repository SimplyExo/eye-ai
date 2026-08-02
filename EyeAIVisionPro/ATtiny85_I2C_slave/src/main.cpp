#ifndef PIO_UNIT_TESTING

#include "LEDController.hpp"
#include "HoldTransistorController.hpp"
#include <Arduino.h>

LEDController * led;
HoldTransistorController * bjt;

void setup() {
  // put your setup code here, to run once:
  bjt = new HoldTransistorController();
  led = new LEDController();  
}

void loop() {
  // put your main code here, to run repeatedly:
  led->set_led(LEDController::GREEN_ON);
  delay(500);
  led->set_led(LEDController::RED_ON);
  delay(500);
}

#endif
