#ifndef PIO_UNIT_TESTING

#include <configuration.h>
#include <LEDController.hpp>
#include <BatteryState.hpp>
#include <HoldTransistorController.hpp>
#include <Arduino.h>
#include <TinyWireS.h>


LEDController led;
HoldTransistorController bjt;
BatteryState battery;


volatile byte receivedCommand = 0;

volatile uint16_t responseValue = 0;
volatile bool responseReady = false;


void receive(uint8_t count);
void send_response();


void setup() {
  TinyWireS.begin(SLAVE_ADDR);

  TinyWireS.onReceive(receive);
  TinyWireS.onRequest(send_response);
}


void receive(uint8_t count) {
  if (TinyWireS.available()) {
    receivedCommand = TinyWireS.receive();
  }
}

void send_response()
{
  if (responseReady) {
    TinyWireS.send(highByte(responseValue));
    TinyWireS.send(lowByte(responseValue));
    responseReady = false;
  }
  else {
    TinyWireS.send(0);
  }
}

void loop()
{
  TinyWireS_stop_check();

  byte command = receivedCommand;

  if (command == 0) {
    return;
  }

  receivedCommand = 0;

  switch (command) {
    case SET_LED_RED:
      led.set_led(LEDController::RED_ON);
      break;


    case SET_LED_GREEN:
      led.set_led(LEDController::GREEN_ON);
      break;

    case DISABLE_TRANSISTOR:
      bjt.set_transistor(
        HoldTransistorController::TRANSISTOR_OFF
      );
      break;

    case READ_BATTERY_STATE:
      responseValue = battery.read_analog();  // 16-Bit Value
      responseReady = true;
      break;
  }
}

#endif