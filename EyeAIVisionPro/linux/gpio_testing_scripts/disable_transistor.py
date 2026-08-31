#!/usr/bin/env python3

from smbus2 import SMBus

I2C_BUS = 1
SLAVE_ADDR = 0x08

DISABLE_TRANSISTOR = 3


def send_command(bus, command):
    try:
        bus.write_byte(SLAVE_ADDR, command)
        return True
    except Exception as e:
        print(f"I2C error: {e}")
        return False


def main():
    with SMBus(I2C_BUS) as bus:
        input("WARNING: THIS WILL CUT OFF THE POWER SUPPLY! Press enter to proceed: ")
        print("Sending command: Disabling transistor")

        if send_command(bus, DISABLE_TRANSISTOR):
            print("Command sent successfully.")
        else:
            print("Command couldn't be sent.")


if __name__ == "__main__":
    main()