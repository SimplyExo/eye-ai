import RPi.GPIO as GPIO
from time import sleep

# Test script to check the functionality of the On-Off Button on the HAT

BUTTON_PIN = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN)

for i in range(30):
    if GPIO.input(BUTTON_PIN):
        state = "Released"
    else:
        state = "Pressed"

    print(f"Button status ({i}): {state}")
    sleep(0.5)

GPIO.cleanup()

