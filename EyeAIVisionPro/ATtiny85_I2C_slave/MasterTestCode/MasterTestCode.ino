#include <Arduino.h>
#include <Wire.h>

#define SLAVE_ADDR 0x08

#define SET_LED_GREEN 1
#define SET_LED_RED 2
#define DISABLE_TRANSISTOR 3
#define READ_BATTERY_STATE 4


void printMenu();
bool sendCommand(byte command);
void red_led();
void green_led();
void disable_transistor();
void read_battery_state();


void setup()
{
  Serial.begin(9600);

  Wire.begin();

  delay(1000);

  Serial.println("I2C Master gestartet");

  printMenu();
}


void loop()
{
  if (Serial.available())
  {
    char incomingByte = Serial.read();

    switch (incomingByte)
    {
      case '0':
        red_led();
        break;

      case '1':
        green_led();
        break;

      case '2':
        read_battery_state();
        break;

      case '3':
        disable_transistor();
        break;

      default:
        Serial.println("Ungueltige Eingabe");
        break;
    }

    printMenu();
  }
}


void printMenu()
{
  Serial.println();
  Serial.println("Attiny85 I2C Tester");
  Serial.println("-------------------");
  Serial.println("0: Rote LED einschalten");
  Serial.println("1: Gruene LED einschalten");
  Serial.println("2: Batteriespannung lesen");
  Serial.println("3: Transistor ausschalten");
  Serial.println();
}


bool sendCommand(byte command)
{
  Wire.beginTransmission(SLAVE_ADDR);
  Wire.write(command);

  byte error = Wire.endTransmission();

  if (error != 0)
  {
    Serial.print("I2C Fehler: ");
    Serial.println(error);
    return false;
  }

  return true;
}


void red_led()
{
  if (sendCommand(SET_LED_RED))
  {
    Serial.println("Rote LED aktiviert");
  }
}


void green_led()
{
  if (sendCommand(SET_LED_GREEN))
  {
    Serial.println("Gruene LED aktiviert");
  }
}


void disable_transistor()
{
  if (sendCommand(DISABLE_TRANSISTOR))
  {
    Serial.println("Transistor deaktiviert");
  }
}


void read_battery_state()
{
  // Kommando senden
  if (!sendCommand(READ_BATTERY_STATE))
  {
    return;
  }


  // ATtiny Zeit geben, ADC zu lesen
  delay(10);


  // Antwort anfordern
  Wire.requestFrom(SLAVE_ADDR, (uint8_t)2);


  if (Wire.available())
  {
    uint8_t high = Wire.read();
    uint8_t low  = Wire.read();

    uint16_t batteryValue = ((uint16_t)high << 8) | low;

    Serial.print("Battery ADC: ");
    Serial.println(batteryValue);
  }
  else
  {
    Serial.println("Keine Antwort vom ATtiny");
  }
}
