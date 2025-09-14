#include "ble.h"

// Service & Characteristic UUIDs
#define GATTS_SERVICE_UUID 0x00FF
#define GATTS_CHAR_UUID    0xFF01
#define GATTS_NUM_HANDLE   4

/*
static uint8_t char_value[20] = "Hallo von ESP32!";
static esp_gatt_char_prop_t char_property = 0;

static void gatts_profile_event_handler(esp_gatts_cb_event_t event,
                                        esp_gatt_if_t gatts_if,
                                        esp_ble_gatts_cb_param_t *param);

static struct gatts_profile_inst {
    esp_gatts_cb_t gatts_cb;
    uint16_t gatts_if;
    uint16_t app_id;
    uint16_t conn_id;
    uint16_t service_handle;
    esp_gatt_srvc_id_t service_id;
} gl_profile = {
    .gatts_cb = gatts_profile_event_handler,
    .gatts_if = ESP_GATT_IF_NONE,
};

// GAP Event Callback
static void gap_event_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param)
{
    switch(event) {
        case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
            esp_ble_gap_start_advertising(&(esp_ble_adv_params_t){
                .adv_int_min        = 0x20,
                .adv_int_max        = 0x40,
                .adv_type           = ADV_TYPE_IND,
                .own_addr_type      = BLE_ADDR_TYPE_PUBLIC,
                .channel_map        = ADV_CHNL_ALL,
                .adv_filter_policy  = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
            });
            break;
        default:
            break;
    }
}

// GATT Server Event Callback
static void gatts_profile_event_handler(esp_gatts_cb_event_t event,
                                        esp_gatt_if_t gatts_if,
                                        esp_ble_gatts_cb_param_t *param)
{
    switch(event) {
        case ESP_GATTS_REG_EVT:
        {
            gl_profile.service_id.is_primary = true;
            gl_profile.service_id.id.inst_id = 0x00;
            gl_profile.service_id.id.uuid.len = ESP_UUID_LEN_16;
            gl_profile.service_id.id.uuid.uuid.uuid16 = GATTS_SERVICE_UUID;

            esp_ble_gatts_create_service(gatts_if, &gl_profile.service_id, GATTS_NUM_HANDLE);
            break;
        }
        case ESP_GATTS_CREATE_EVT:
            gl_profile.service_handle = param->create.service_handle;

            esp_ble_gatts_start_service(gl_profile.service_handle);

            esp_ble_gatts_add_char(gl_profile.service_handle, 
                                   &(esp_bt_uuid_t){.len = ESP_UUID_LEN_16, .uuid.uuid16 = GATTS_CHAR_UUID},
                                   ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
                                   ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_WRITE,
                                   char_value, NULL);
            break;
        case ESP_GATTS_WRITE_EVT:
            if(param->write.len <= sizeof(char_value)) {
                memcpy(char_value, param->write.value, param->write.len);
                char_value[param->write.len] = 0; // Null-terminator
                ESP_LOGI(GATTS_TAG, "Characteristic geschrieben: %s", char_value);
            }
            break;
        default:
            break;
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_BLE));

    ESP_ERROR_CHECK(esp_bluedroid_init());
    ESP_ERROR_CHECK(esp_bluedroid_enable());

    ESP_ERROR_CHECK(esp_ble_gap_register_callback(gap_event_handler));
    ESP_ERROR_CHECK(esp_ble_gatts_register_callback(gatts_profile_event_handler));
    ESP_ERROR_CHECK(esp_ble_gatts_app_register(0));
}
*/