#pragma once

#include <driver/touch_pad.h>
#include "driver/gpio.h"

/* Hardware Config */
#define RED_LED_GPIO    12
#define GREEN_LED_GPIO  13
#define FLASHLIGHT      4

#define TOUCH_PAD_GPIO14_CHANNEL TOUCH_PAD_NUM6
#define TOUCH_THRESHOLD 800