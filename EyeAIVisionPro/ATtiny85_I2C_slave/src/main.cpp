#ifndef PIO_UNIT_TESTING

#include <Arduino.h>
#include <TinyWireS.h>
#include <configuration.h>
#include <LEDController.hpp>
#include <BatteryState.hpp>
#include <HoldTransistorController.hpp>

LEDController led;
HoldTransistorController bjt;
BatteryState battery;

volatile byte lastCommand = 0;

void receive(uint8_t count);
void send_response();

void setup() {
  delay(250);   // avoid bouncing
  battery.init();
  TinyWireS.begin(SLAVE_ADDR);

  TinyWireS.onReceive(receive);
  TinyWireS.onRequest(send_response);
}

void receive(uint8_t count) {
  while (TinyWireS.available()) {
    byte command = TinyWireS.receive();

    // Nur gültige Kommandos zwischenspeichern
    if (command == SET_LED_RED || command == SET_LED_GREEN || 
        command == DISABLE_TRANSISTOR || command == READ_BATTERY_STATE) {
      lastCommand = command;
    }

    // Steuerbefehle (LED, Transistor) direkt beim Empfang ausführen
    switch (command) {
      case SET_LED_RED:
        led.set_led(LEDController::RED_ON);
        break;

      case SET_LED_GREEN:
        led.set_led(LEDController::GREEN_ON);
        break;

      case DISABLE_TRANSISTOR:
        bjt.set_transistor(HoldTransistorController::TRANSISTOR_OFF);
        break;
    }
  }
}

void send_response()
{
  // Prüfen, ob der letzte Befehl eine Batterie-Abfrage war
  if (lastCommand == READ_BATTERY_STATE) {
    byte adcValue = (byte)battery.read_analog();
    TinyWireS.send(adcValue);

    lastCommand = 0; 
  } else {
    TinyWireS.send(0);
  }
}

void loop()
{
  TinyWireS_stop_check();
}

#endif