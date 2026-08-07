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

### On your local machine
3. Copy public key to your development machine
```shell
ssh-copy-id eyeai@192.168.4.1       # Make sure you are connected to the RPI hotspot!
```