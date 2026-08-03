## ATtiny85 I2C Slave

### Purpouse
Due to the Raspberry Pi Zero W not having a built-in ADC as well as the ability to set GPIO pins on HIGH immediately after being started, it was necessary to add another small microcontroller to our Raspberry Pi HAT. In our case it is the ATtiny85, which takes the role of controlling the status LED, keeping the power supply alive and measuring the battery voltage (in order to estimate the state of charge).

### Known commands

| Command  |   Code   | Response |
|----------|----------|----------|
| Set green status LED    | 0x01   | Nothing   |
| Set red status LED    | 0x02   | Nothing   |
| Turn off transistor    | 0x03   | Nothing   |
| Measure battery voltage    | 0x04   | Raw 8-Bit ADC value  |
