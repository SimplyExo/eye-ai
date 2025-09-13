#include "wifi_sta.h"

EventGroupHandle_t s_wifi_event_group;
const char *TAG_WIFI = "WiFi";
int s_retry_num = 0;

void event_handler_sta(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num > 5) {
            wifi_init_softap(); // TODO: FIX THIS
        } else {
            gpio_set_level(GREEN_LED_GPIO, 0); // Erst LED ausschalten, dann andere einschalten, damit Last an Widerstand (verbunden an GND) nicht zu hoch wird
            gpio_set_level(RED_LED_GPIO, 1);
            esp_wifi_connect();

            ESP_LOGI(TAG_WIFI, "retry to connect to the AP");
            ESP_LOGI(TAG_WIFI,"connect to the AP fail");
            //s_retry_num++;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        s_retry_num = 0;
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG_WIFI, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        gpio_set_level(RED_LED_GPIO, 0);
        gpio_set_level(GREEN_LED_GPIO, 1);
    }
}

void init_sta(char * ssid, char * passwd)//, char * ip, char * gw)
{
    char ssid_buff[32];
    char passwd_buff[32];

    strncpy(ssid_buff, ssid, sizeof(ssid_buff)-1);
    ssid_buff[sizeof(ssid_buff)-1] = 0;

    strncpy(passwd_buff, passwd, sizeof(passwd_buff)-1);
    passwd_buff[sizeof(passwd_buff)-1] = 0;

    ESP_LOGI(TAG_WIFI, "SSID: %s, PASSWD: %s", ssid_buff, passwd_buff);

//    ESP_ERROR_CHECK(esp_wifi_stop());

    if (CONFIG_LOG_MAXIMUM_LEVEL > CONFIG_LOG_DEFAULT_LEVEL) {
        esp_log_level_set("wifi", CONFIG_LOG_MAXIMUM_LEVEL);
    }

    s_wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());

    // Default WiFi STA interface
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();

    // Stoppe DHCP client (falls statische IP)
    //esp_netif_dhcpc_stop(sta_netif);

    // WiFi initialisieren
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // Event-Handler registrieren
    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler_sta,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler_sta,
                                                        NULL,
                                                        &instance_got_ip));

    // WiFi Config
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = {0},
            .password = {0},
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,  // WPA2 bevorzugen, WPA3 nur wenn unterstützt
            .sae_pwe_h2e = WPA3_SAE_PWE_BOTH          // WPA3 H2E fallback
        },
    };

    // Statische IP setzen
    /*esp_netif_ip_info_t ip_info;
    ip_info.ip.addr = ipaddr_addr(ip);
    ip_info.gw.addr = ipaddr_addr(gw);
    ip_info.netmask.addr = ipaddr_addr("255.255.255.0");
    ESP_ERROR_CHECK(esp_netif_set_ip_info(sta_netif, &ip_info));*/

    // WiFi Credentials kopieren
    strncpy((char*)wifi_config.sta.ssid, ssid_buff, sizeof(wifi_config.sta.ssid)-1);
    strncpy((char*)wifi_config.sta.password, passwd_buff, sizeof(wifi_config.sta.password)-1);

    // Station Mode setzen
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));

    // Power-Save deaktivieren
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    // Start WiFi
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG_WIFI, "wifi_init_sta finished.");

    // Warten bis verbunden oder fehlgeschlagen
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG_WIFI, "Connected to AP SSID:%s password:%s",
                 ssid_buff, passwd_buff);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGW(TAG_WIFI, "Failed to connect to SSID:%s, password:%s",
                 ssid_buff, passwd_buff);

        // Optional: reconnect retry
        vTaskDelay(pdMS_TO_TICKS(5000));  // 5 Sekunden warten
        //init_sta(ssid, passwd, ip, gw);   // rekursiv reconnecten
    } else {
        ESP_LOGE(TAG_WIFI, "UNEXPECTED EVENT");
    }
}