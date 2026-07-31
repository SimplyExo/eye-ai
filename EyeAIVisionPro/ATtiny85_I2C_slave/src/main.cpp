#include <Arduino.h>

void setup() {
  // put your setup code here, to run once:
  pinMode(1, OUTPUT);   // Onboard-LED
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(1, HIGH); // LED on
  delay(500);           // delay

  digitalWrite(1, LOW);  // LED off
  delay(500);           // delay
}
