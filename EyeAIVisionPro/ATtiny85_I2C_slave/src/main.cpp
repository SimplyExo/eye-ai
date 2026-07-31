#include "LEDController.hpp"
#include <Arduino.h>

LEDController * led;

void setup() {
  // put your setup code here, to run once:
  led = new LEDController();
  led->init();
}

void loop() {
  // put your main code here, to run repeatedly:
  led->set_led(LEDController::GREEN_ON);
  delay(500);
  led->set_led(LEDController::RED_ON);
  delay(500);
}
