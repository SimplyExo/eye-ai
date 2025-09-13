#pragma once
#include <string.h>
#include "esp_system.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "lwip/ip_addr.h"
#include "esp_netif_ip_addr.h"
#include "wifi_ap.h"
#include "pinout.h"

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

#define WIFI_SSID   "EyeAIApp"
#define WIFI_PASSWD "eyeaiapphotspot"

#define ESP_WIFI_SCAN_AUTH_MODE_THRESHOLD WIFI_AUTH_WPA_WPA2_PSK
#define ESP_WIFI_SAE_MODE WPA3_SAE_PWE_BOTH
#define EXAMPLE_H2E_IDENTIFIER ""

void event_handler(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data);

void init_sta(char * ssid, char * passwd); //, char * ip, char * gw);
