#pragma once 

// Pinout
#define SDA_PIN 0
#define SCL_PIN 2
#define HOLD_TRANSISTOR 1
#define LED_GREEN_ENABLE 4
#define BATTERY_VOLTAGE 3

// I2C
#define SLAVE_ADDR 0x08

#define SET_LED_GREEN 1
#define SET_LED_RED 2
#define DISABLE_TRANSISTOR 3
#define READ_BATTERY_STATE 4
