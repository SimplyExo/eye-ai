#!/usr/bin/env python3

from smbus2 import SMBus

from time import sleep

I2C_BUS = 1
SLAVE_ADDR = 0x08

COMMAND = 0x04

def read_adc_val(bus: SMBus):
    # Send command
    bus.write_byte(SLAVE_ADDR, COMMAND)

    sleep(0.01) # give slave time to answer

    # Read both bytes
    data = bus.read_i2c_block_data(SLAVE_ADDR, 0, 2)

    high = data[0]
    low = data[1]

    # Merge bytes into a 16-bit integer
    value = (high << 8) | low

    return value
    

def main():
    with SMBus(I2C_BUS) as bus:
        try:
            print(f"ADC value: {read_adc_val(bus)}")

        except KeyboardInterrupt:
            print("\nTest beendet.")


if __name__ == "__main__":
    main()
