#!/usr/bin/env python3

import time
from smbus2 import SMBus

I2C_BUS = 1
SLAVE_ADDR = 0x08

SET_LED_GREEN = 1
SET_LED_RED = 2


def send_command(bus, command):
    try:
        bus.write_byte(SLAVE_ADDR, command)
        return True
    except Exception as e:
        print(f"I2C error: {e}")
        return False


def main():
    with SMBus(I2C_BUS) as bus:
        print("Starting LED test...")

        try:
            while True:
                print("Green")
                send_command(bus, SET_LED_GREEN)
                time.sleep(1)

                print("Rot")
                send_command(bus, SET_LED_RED)
                time.sleep(10)

        except KeyboardInterrupt:
            print("\nTest beendet.")


if __name__ == "__main__":
    main()