#include "tcp_server.h"

static const char *TAG = "TCP";

int keepAlive = 1;
int keepIdle = 5;
int keepInterval = 5;
int keepCount = 3;
int flag = 1;

bool touch_tapped() {
    uint16_t value = 0;
    touch_pad_read_raw_data(TOUCH_PAD_GPIO14_CHANNEL, &value);
    return value < TOUCH_THRESHOLD;
}

void tcp_server_task(void *pvParameters)
{
    int addr_family = AF_INET;
    int ip_protocol = IPPROTO_IP;

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(PORT);

    int listen_sock = socket(addr_family, SOCK_STREAM, ip_protocol);
    if (listen_sock < 0) {
        ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Socket created");

    int err = bind(listen_sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
    if (err != 0) {
        ESP_LOGE(TAG, "Socket unable to bind: errno %d", errno);
        close(listen_sock);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Socket bound, port %d", PORT);

    err = listen(listen_sock, 1);
    if (err != 0) {
        ESP_LOGE(TAG, "Error occurred during listen: errno %d", errno);
        close(listen_sock);
        vTaskDelete(NULL);
        return;
    }

    while (1) {
        ESP_LOGI(TAG, "Waiting for a client...");
        struct sockaddr_in6 source_addr;
        socklen_t addr_len = sizeof(source_addr);
        int sock = accept(listen_sock, (struct sockaddr *)&source_addr, &addr_len);

        setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &keepAlive, sizeof(int));
        setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, &keepIdle, sizeof(int));
        setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &keepInterval, sizeof(int));
        setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, &keepCount, sizeof(int));
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int));

        if (sock < 0) {
            ESP_LOGE(TAG, "Unable to accept connection: errno %d", errno);
            break;
        }
        ESP_LOGI(TAG, "Client connected");

        while (0) {
            int sent = 0; //send(sock, msg, strlen(msg), 0);
            if (sent < 0) {
                ESP_LOGE(TAG, "Error sending: errno %d", errno);
                break; // Verbindung verloren
            }
            vTaskDelay(1000 / portTICK_PERIOD_MS); // 1 Sekunde warten
        }

        while (1) {
            int taps = 0;

            if (touch_tapped()) {
                taps++;
                long timer = esp_timer_get_time() + TAP_THRESHOLD;
                while (touch_tapped()) { vTaskDelay(1); }
                
                while (esp_timer_get_time() < timer)
                {
                    if (touch_tapped()) {
                        taps++;
                        while (touch_tapped()) { vTaskDelay(1); }
                    }
                }
            }
            
            if (taps > 0) {
                if (taps > 2) {
                    taps = 2;
                }

                char buffer[20];
                int len = sprintf(buffer, "%d", taps);  // len = Anzahl der geschriebenen Zeichen

                int sent = send(sock, buffer, len, 0);   // alle Zeichen senden, aber NICHT das '\0'

                ESP_LOGI(TAG, "Sent data: %s", buffer);
            }

            char tmp;
            int ret = recv(sock, &tmp, 1, MSG_PEEK | MSG_DONTWAIT);
            if (ret == 0) {
                ESP_LOGW(TAG, "Client disconnected");
                close(sock);
                break;
            }

            vTaskDelay(pdMS_TO_TICKS(10));
        }

        shutdown(sock, 0);
        close(sock);
        ESP_LOGI(TAG, "Client disconnected");
    }

    close(listen_sock);
    vTaskDelete(NULL);
}

void start_tcp_server(void)
{
    xTaskCreate(tcp_server_task, "tcp_server", 4096, (void*)AF_INET, 5, NULL);
}