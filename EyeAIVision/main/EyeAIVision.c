#include "pinout.h"
#include "camera.h"
#include "wifi_sta.h"
#include "http_server.h"
#include "tcp_server.h"
#include "ble.h"

void init_hw()
{
    gpio_reset_pin(RED_LED_GPIO);
    gpio_reset_pin(GREEN_LED_GPIO);
    gpio_reset_pin(FLASHLIGHT);

    gpio_set_direction(RED_LED_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_direction(GREEN_LED_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_direction(FLASHLIGHT, GPIO_MODE_OUTPUT);

    gpio_set_level(RED_LED_GPIO, 1);
    gpio_set_level(GREEN_LED_GPIO, 0);
    gpio_set_level(FLASHLIGHT, 0);

    touch_pad_init();
    touch_pad_set_voltage(TOUCH_HVOLT_2V7, TOUCH_LVOLT_0V5, TOUCH_HVOLT_ATTEN_1V);
    touch_pad_config(TOUCH_PAD_GPIO14_CHANNEL, -1);
    touch_pad_filter_start(10);
}

void app_main(void)
{
    init_hw();
    init_camera();

    init_sta();
    startCameraServer();
    start_tcp_server();
}
