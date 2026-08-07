# GPIO testing scripts
Basic scripts for testing hardware functionality of:
- On-Off Button
- Status LED (via I2C)
- Hold transistor (via I2C)
- State of charge (via I2C)

## Setup
### On the Raspberry Pi
1. Generate SSH keypair:
```shell 
ssh-keygen
```

2. Create a directory for script files being executed during testing
```shell
mkdir ~/gpio_testing_scripts
```

3. Install utilities and libraries for i2c bus
```shell
sudo apt-get install i2c-tools
sudo apt-get install python3-smbus
```

4. Reboot
```shell
sudo reboot
```

5. Verify whether the ATtiny85 I2C Slave gets recognized
```shell
i2cdetect -y 1
```

This command should output something like this, if the ATtiny85 gets recognized at address 0x08
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         08 -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
70: -- -- -- -- -- -- -- --
```

### On your local machine
6. Copy public key to your development machine. The standard Password is ```vision```
```shell
ssh-copy-id eyeai@192.168.4.1       # Make sure you are connected to the RPI hotspot!
```

7. (Optional, but recommended) Verify whether your SSH client accepts the new public key
```ssh
ssh eyeai@192.168.4.1
```