#include "pinout.h"
#include "http_server.h"
#include "wifi_ap.h"

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
    //Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    init_hw();

    // Default Event Loop nur einmal erstellen
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    init_camera();
    init_sta("EyeAI", "123456789");
    startHTTPServer();
    
    // STA-Mode wird nach HTTP request mit Andmeldedaten gestartet!
}
